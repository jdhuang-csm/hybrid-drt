import numpy as np
from scipy.special import gamma, loggamma, erf
# from scipy.interpolate import interp1d


def harmonic_mean(x, y):
    return 2 * x * y / (x + y)


def pdf_normal(x, loc, scale):
    return 1 / (scale * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - loc) ** 2 / scale ** 2)


def log_pdf_normal(x, mu, sigma):
    return -0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma) + ((x - mu) / sigma) ** 2)


def cdf_normal(x, loc, scale):
    return 0.5 * (1 + erf((x - loc) / (scale * (2 ** 0.5))))


def pdf_gamma(x, shape, rate):
    return (rate ** shape / gamma(shape)) * x ** (shape - 1) * np.exp(-rate * x)


def log_pdf_gamma(x, alpha, beta, include_constants=True):
    lp = (alpha - 1) * np.log(x) - beta * x
    if include_constants:
        lp += alpha * np.log(beta) - loggamma(alpha)
    return lp


def pdf_invgamma(x, alpha, beta):
    return (beta ** alpha / gamma(alpha)) * x ** (-alpha - 1) * np.exp(-beta / x)


def pdf_exp(x, rate):
    return rate * np.exp(-rate * x)


def pdf_laplace(x, mu, rate):
    return 0.5 * rate * np.exp(-rate * np.abs(x - mu))


def pdf_dirichlet(x, alpha, include_constants=True):
    pdf = np.prod(x ** (alpha - 1))

    if include_constants:
        beta = np.prod(gamma(alpha)) / gamma(np.sum(alpha))
        pdf /= beta

    return pdf


def log_pdf_dirichlet(x, alpha, include_constants=True):
    # shape = np.shape(x)
    x = np.atleast_2d(x)
    alpha = np.atleast_2d(alpha)

    lp = np.sum((alpha - 1) * np.log(x), axis=1)

    if include_constants:
        lb = np.sum(loggamma(alpha), axis=1) - loggamma(np.sum(alpha, axis=1))
        lp -= lb

    # if len(shape) == 1:
    #     lp =

    return lp.flatten()


def pdf_beta(x, alpha, beta, include_constants=True):
    pdf = x ** (alpha - 1) * (1 - x) ** (beta - 1)

    if include_constants:
        beta = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
        pdf /= beta

    return pdf


def log_pdf_beta(x, alpha, beta, include_constants=True):
    lp = (alpha - 1) * np.log(x) + (beta - 1) * np.log(1 - x)

    if include_constants:
        lb = loggamma(alpha) + loggamma(beta) - loggamma(alpha + beta)
        lp -= lb


def pdf_lognormal(x, mu, sigma):
    return np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2) / (x * sigma * np.sqrt(2 * np.pi))


def std_normal_quantile(quantiles):
    """Get value of standard normal random variable corresponding to quantiles"""
    s_interp = np.linspace(-7, 7, 2000)
    cdf = cdf_normal(s_interp, 0, 1)
    s = np.interp(quantiles, cdf, s_interp)

    return s


def robust_std(x):
    """Estimate standard deviation from interquartile range"""
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)

    return (q3 - q1) / 1.349


def bic(k, n, llh):
    """
    Bayesian information criterion
    :param int k: Number of parameters
    :param int n: Number of data points
    :param float llh: Maximized log-likelihood
    :return:
    """
    return k * np.log(n) - 2 * llh


def bayes_factor(c1, c2, criterion='bic'):
    if criterion == 'bic':
        return np.exp(-0.5 * (c1 - c2))
    elif criterion in ('lml', 'lml-bic'):
        return np.exp(c1 - c2)
    else:
        raise ValueError(f'Invalid criterion {criterion}')


def norm_bayes_factors(crit_values, criterion='bic'):
    if criterion == 'bic':
        best_value = np.min(crit_values)
        return np.exp(-0.5 * (crit_values - best_value))
    elif criterion in ('lml', 'lml-bic'):
        best_value = np.max(crit_values)
        return np.exp(crit_values - best_value)
    else:
        raise ValueError(f'Invalid criterion {criterion}')

