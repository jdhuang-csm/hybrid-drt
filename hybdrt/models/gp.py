import numpy as np
import torch
import gpytorch
from gpytorch.lazy import KroneckerProductLazyTensor, MatmulLazyTensor, lazify, delazify
from ..utils import stats


class Normalizer:
    def __init__(self, center=True, scale=False, normalize_by_var=False):
        self.center = center
        self.scale = scale
        self.normalize_by_var = normalize_by_var

        self.raw_mean = None
        self.raw_scale = None

    def fit(self, x):
        if self.center:
            if self.normalize_by_var:
                self.raw_mean = torch.mean(x, dim=0)
            else:
                self.raw_mean = torch.mean(x)
        else:
            self.raw_mean = 0

        if self.scale:
            if self.normalize_by_var:
                self.raw_scale = torch.std(x, dim=0)
            else:
                self.raw_scale = torch.std(x)
        else:
            self.raw_scale = 1

    def transform(self, x):
        x_norm = (x - self.raw_mean) / self.raw_scale
        return x_norm

    def inverse_transform(self, x_norm):
        x = x_norm * self.raw_scale + self.raw_mean
        return x


class BaseGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks,
                 center_x=False, scale_x=False, normalize_x_by_var=True,
                 center_y=False, scale_y=False, normalize_y_by_var=False):
        # Convert input data to tensors
        train_x = torch.Tensor(train_x)
        train_y = torch.Tensor(train_y)

        # Define data normalization
        self.x_norm = Normalizer(center=center_x, scale=scale_x, normalize_by_var=normalize_x_by_var)
        self.y_norm = Normalizer(center=center_y, scale=scale_y, normalize_by_var=normalize_y_by_var)
        self.x_norm.fit(train_x)
        self.y_norm.fit(train_y)
        train_x = self.x_norm.transform(train_x)
        train_y = self.y_norm.transform(train_y)

        # Store training data
        self.train_x = train_x
        self.train_y = train_y

        self.num_tasks = num_tasks

        # Initialize ExactGP
        super().__init__(train_x, train_y, likelihood)

    def update_train_data(self, train_x, train_y, strict=True, update_normalization=False):
        # Convert input data to tensors
        train_x = torch.Tensor(train_x)
        train_y = torch.Tensor(train_y)

        # Update and apply data normalization
        if update_normalization:
            self.x_norm.fit(train_x)
            self.y_norm.fit(train_y)
        train_x = self.x_norm.transform(train_x)
        train_y = self.y_norm.transform(train_y)

        # Store training data
        self.train_x = train_x
        self.train_y = train_y

    def optimize_hypers(self, iterations=50, lr=0.1, verbose=False):
        # Switch to train mode
        self.train()
        self.likelihood.train()

        # Define optimizer and marginal log-likelihood
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        print_interval = int(iterations / 5)

        for i in range(iterations):
            optimizer.zero_grad()
            output = self(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            if verbose and (i % print_interval == 0 or i == iterations - 1):
                print('Iter %d/%d - Loss: %.3f' % (i + 1, iterations, loss.item()))
            optimizer.step()

    def predict(self, test_x, fast_pred=True, quantiles=None):
        mvn = self.evaluate_posterior_mvn(test_x, fast_pred)

        if quantiles is not None:
            s_quant = torch.Tensor(stats.std_normal_quantile(quantiles))
            mean = mvn.mean

            # Format output array: first dimension is quantile dimension, remaining dimensions are dimensions of mean
            if self.num_tasks > 1:
                y_pred = torch.tile(mean, (len(quantiles), 1, 1)) + (
                        torch.tile(torch.sqrt(mvn.variance), (len(quantiles), 1, 1))
                        * torch.tile(s_quant, (mean.shape[1], mean.shape[0], 1)).T
                )
            else:
                y_pred = torch.tile(mean, (len(quantiles), 1)) + (
                        torch.tile(torch.sqrt(mvn.variance), (len(quantiles), 1))
                        * torch.tile(s_quant, (len(mean), 1)).T
                )
        else:
            y_pred = mvn.mean

        # Perform inverse normalization
        if quantiles is not None:
            for i in range(y_pred.shape[0]):
                y_pred[i] = self.y_norm.inverse_transform(y_pred[i])
        else:
            y_pred = self.y_norm.inverse_transform(y_pred)

        return y_pred.detach().numpy()

    def evaluate_posterior_mvn(self, test_x, fast_pred=True):
        # Switch to eval mode
        self.eval()
        self.likelihood.eval()

        # Convert test_x to tensor
        test_x = torch.Tensor(test_x)

        # Transform test_x based on norm
        test_x = self.x_norm.transform(test_x)

        if fast_pred:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                mvn = self.likelihood(self(test_x))
        else:
            mvn = self.likelihood(self(test_x))

        return mvn

    def compute_precision(self, x, remove_scaling=True, add_variance=None, multiply_variance=None):
        # Get posterior MVN (disable fast pred to ensure covar matrix is invertible)
        mvn = self.evaluate_posterior_mvn(x, fast_pred=False)

        covariance_matrix = mvn.covariance_matrix

        # Remove target scaling
        if remove_scaling:
            if self.y_norm.normalize_by_var:
                scale_diag = torch.diag(self.y_norm.raw_scale)
            else:
                scale_diag = torch.eye(covariance_matrix.shape[0]) * self.y_norm.raw_scale
            covariance_matrix = torch.matmul(scale_diag, torch.matmul(covariance_matrix, scale_diag))

        # Multiply variance by factor
        if multiply_variance:
            var = torch.diag(covariance_matrix)
            covariance_matrix += torch.diag(var * (multiply_variance - 1))

        # Add variance
        if add_variance is not None:
            if np.shape(add_variance) == ():
                var_diag = torch.eye(covariance_matrix.shape[0])
            else:
                var_diag = torch.diag(add_variance)
            covariance_matrix += var_diag

        # Invert via Cholesky factorization
        unbroadcasted_scale_tril = torch.linalg.cholesky(covariance_matrix)

        return torch.cholesky_inverse(unbroadcasted_scale_tril).expand(
            mvn._batch_shape + mvn._event_shape + mvn._event_shape)


class SimpleGP(BaseGP):
    def __init__(self, train_x, train_y, likelihood, zero_mean=False, **kw):
        # Initialize BaseGP
        super().__init__(train_x, train_y, likelihood, 1, **kw)

        # Create mean and covariance modules
        if zero_mean:
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=self.train_x.shape[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)  # prior mean vector
        covar_x = self.covar_module(x)  # prior covariance matrix
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class IndependentMultitaskGP(BaseGP):
    def __init__(self, train_x, train_y, likelihood, center_x=False, scale_x=False,
                 center_y=False, scale_y=False):
        # Initialize BaseGP
        super().__init__(train_x, train_y, likelihood, 1, center_x, scale_x, center_y, scale_y)

        # Get batch shape (num tasks)
        batch_shape = torch.Size([train_y.shape[1]])

        # Create mean and covariance modules
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
            batch_shape=batch_shape
        )

    def forward(self, x):
        mean_x = self.mean_module(x)  # prior mean vector
        covar_x = self.covar_module(x)  # prior covariance matrix
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class MultitaskGP(BaseGP):
    def __init__(self, train_x, train_y, likelihood, covar_module, zero_mean=False, **kw
                 ):
        # Determine number of tasks from data
        num_tasks = train_y.shape[1]

        # Initialize BaseGP
        super().__init__(train_x, train_y, likelihood, num_tasks, **kw)

        # Create mean and covariance modules
        if zero_mean:
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ZeroMean(), num_tasks=num_tasks
            )
        else:
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=num_tasks
            )

        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)  # prior mean vector
        covar_x = self.covar_module(x)  # prior covariance matrix
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)



    # def compute_precision(self, x, data_scales=None, unscale=True):
    #     """
    #     Compute precision matrix via Kronecker product of individual inverse covariance matrices
    #     :param x1:
    #     :param x2:
    #     :return:
    #     """
    #     # I think this is the prior? need the posterior
    #     x = torch.Tensor(x)
    #
    #     # Get individual covariance matrices
    #     data_cov = self.covar_module.data_covar_module.forward(x, x)
    #     task_cov = self.covar_module.task_covar_module.covar_matrix
    #
    #     # Apply data scaling
    #     # Note: data_cov is a correlation matrix. all scaling seems to be handled in task_covar_matrix
    #     if data_scales is not None:
    #         if len(data_scales) != x.shape[0]:
    #             raise ValueError('scales must be same length as x.shape[0]')
    #         else:
    #             data_scales = lazify(torch.Tensor(data_scales))
    #             data_cov = MatmulLazyTensor(data_scales, data_cov)
    #             data_cov = MatmulLazyTensor(data_cov, data_scales)
    #
    #     # Remove task scaling
    #     if unscale and self.y_norm.scale:
    #         if self.y_norm.normalize_by_var:
    #             task_scales = lazify(self.y_norm.raw_scale)
    #             task_cov = MatmulLazyTensor(task_scales, task_cov)
    #             task_cov = MatmulLazyTensor(task_cov, task_scales)
    #         else:
    #             task_cov = task_cov * self.y_norm.raw_scale ** 2
    #
    #     # Delazify for inverse
    #     data_cov = delazify(data_cov)
    #     task_cov = delazify(task_cov)
    #
    #     # Invert individual covariance matrices
    #     data_inv = lazify(torch.inverse(data_cov))
    #     task_inv = lazify(torch.inverse(task_cov))
    #
    #     # Compute precision via Kronecker product
    #     precision = KroneckerProductLazyTensor(data_inv, task_inv)
    #
    #     return precision.numpy()


class GenericMultitaskKernel(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(self, data_covar_module, task_covar_module, num_tasks, **kwargs):
        super().__init__(**kwargs)

        # self.task_covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(batch_shape=self.batch_shape),
        #     batch_shape=self.batch_shape
        # )
        self.task_covar_module = task_covar_module
        self.data_covar_module = data_covar_module
        self.num_tasks = num_tasks

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("MultitaskRBFKernel does not accept the last_dim_is_batch argument.")

        # Get task covariance matrix
        task_index = torch.arange(0, self.num_tasks, dtype=int).reshape([self.num_tasks, 1])
        covar_i = lazify(self.task_covar_module.forward(task_index, task_index, **params))

        # Get data covariance matrix
        covar_x = lazify(self.data_covar_module.forward(x1, x2, **params))

        res = KroneckerProductLazyTensor(covar_x, covar_i)
        return res.diag() if diag else res

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks



