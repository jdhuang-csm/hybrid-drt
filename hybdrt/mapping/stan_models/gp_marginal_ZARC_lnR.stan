functions {
	vector HN_gamma(int M, vector tau, real tau_0, real alpha, real beta) {
		vector[M] tt0 = tau ./ tau_0; // precalculate for efficiency
		vector[M] gamma_out;
		for (m in 1:M) {
			real theta = atan2(sin(pi() * beta), (pow(tt0[m], beta) + cos(pi() * beta)));
			real nume = (1 / pi()) * pow(tt0[m], beta * alpha) * sin(alpha * theta);
			real deno = pow(1 + 2 * cos(pi() * beta) * pow(tt0[m], beta) + pow(tt0[m], 2 * beta), alpha / 2);
			gamma_out[m] = nume / deno;
		}
		return gamma_out;
	}
	vector ZARC_gamma(vector tau, real tau_0, real beta) {
	    int M = size(tau);
	    vector[M] tt0 = tau ./ tau_0; // precalculate for convenience
		vector[M] gamma_out;
		for (m in 1:M) {
			real nume = sin((1 - beta) * pi());
			real deno = 2 * pi() * (cosh(beta * log(tt0[m])) - cos((1 - beta) * pi()));
			gamma_out[m] = nume / deno;
		}
		return gamma_out;
	}
	vector raw_to_actual(vector x_raw, real mu, real scale) {
	    return mu + x_raw * scale;
	}
	vector gp_pred_rng(array[] vector x2,
                     vector y1,
                     array[] vector x1,
                     real alpha,
                     real rho,
                     real sq_sigma,
                     real delta) {
    int N1 = rows(y1);
    int N2 = size(x2);
    vector[N2] f2;
    {
        matrix[N1, N1] L_K;
        vector[N1] K_div_y1;
        matrix[N1, N2] k_x1_x2;
        matrix[N1, N2] v_pred;
        vector[N2] f2_mu;
        matrix[N2, N2] cov_f2;
        matrix[N2, N2] diag_delta;
        matrix[N1, N1] K;
        K = gp_exp_quad_cov(x1, alpha, rho);
        for (n in 1:N1) {
            K[n, n] = K[n, n] + sq_sigma;
        }
        L_K = cholesky_decompose(K);
        K_div_y1 = mdivide_left_tri_low(L_K, y1);
        K_div_y1 = mdivide_right_tri_low(K_div_y1', L_K)';
        k_x1_x2 = gp_exp_quad_cov(x1, x2, alpha, rho);
        f2_mu = (k_x1_x2' * K_div_y1);
        v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
        cov_f2 = gp_exp_quad_cov(x2, alpha, rho) - v_pred' * v_pred;
        diag_delta = diag_matrix(rep_vector(delta, N2));

        f2 = multi_normal_rng(f2_mu, cov_f2 + diag_delta);
    }
    return f2;
    }
}
data {
    int<lower=1> N;
    int<lower=1> D; // x dimensions
    int<lower=1> M; // tau grid size
    int<lower=1> K; // number of discrete elements
    vector<lower=0>[M] tau;
    array[N] vector[D] x;
    array[N] vector[M] y;
    real<lower=0> sigma_gp_scale;
    real<lower=0> sigma_rel_y_scale;
    real<lower=0> sigma_rel_Rp_scale;
    vector[K] R_sign;
}
transformed data {
    array[K] vector[N] mu; // GP mean
    vector<lower=0>[N] Rp; // Polarization resistance

    for (k in 1:K) {
        mu[k] = rep_vector(0.0, N);
    }
    for (n in 1:N) {
        Rp[n] = sum(fabs(y[n]));
    }
}
parameters {
    // TODO: implement ARD (anisotropic kernel)
    vector<lower=0>[K] rho;  // cov length scale
    vector<lower=0>[K] alpha; // discrete element cov magnitude
    vector<lower=0>[K] sigma; // discrete element noise level

    // ZARC parameters
    array[K] vector[N] lnR_raw;
    array[K] vector[N] lntau_raw;
    array[K] vector[N] beta_trans_raw;
    vector[K] lnR_mu;
    vector[K] lntau_mu;
    vector<lower=0>[K] lnR_scale;
    vector<lower=0>[K] lntau_scale;
    vector[K] beta_trans_mu;
    vector<lower=0>[K] beta_trans_scale;
    real<lower=0> sigma_rel_y_raw; // relative surface noise level
    real<lower=0> sigma_rel_Rp_raw;
    // vector<lower=0>[K] alpha_lt;
    // vector<lower=0>[K] sigma_lt;
    // vector<lower=0>[K] alpha_beta;
    // vector<lower=0>[K] sigma_beta;
}
transformed parameters {
    array[K] vector[N] lnR;
    array[K] vector[N] R;
    array[K] vector[N] lntau;
    array[K] vector[N] beta_trans;
    array[K] vector<lower=0, upper=1>[N] beta;
    //vector<lower=0>[N] Rp_hat;
    //vector<lower=0>[N] sigma_y;
    array[N] vector<lower=0>[M] sigma_y;
    vector<lower=0>[N] sigma_Rp;
    vector<lower=0>[K] sq_sigma = square(sigma_gp_scale * sigma);

    // model surface
    array[N] vector[M] y_hat;
    vector[N] Rp_hat;

    //Rp_hat = rep_vector(0.0, N);

    // Transform raw RQ parameters to actual values
    for (k in 1:K) {
        lnR[k] = raw_to_actual(lnR_raw[k], lnR_mu[k], lnR_scale[k]);
        R[k] = exp(lnR[k]) * R_sign[k];
        lntau[k] = raw_to_actual(lntau_raw[k], lntau_mu[k], lntau_scale[k]);
        beta_trans[k] = raw_to_actual(beta_trans_raw[k], beta_trans_mu[k], beta_trans_scale[k]);
        beta[k] = exp(beta_trans[k]) ./ (exp(beta_trans[k]) + 1.0);
    }

    // calculate model surface
    for (n in 1:N) {
        y_hat[n] = rep_vector(0.0, M);
        for (k in 1:K) {
            y_hat[n] = y_hat[n] + R[k][n] * ZARC_gamma(tau, exp(lntau[k][n]), beta[k][n]);
        }
        Rp_hat[n] = sum(fabs(y_hat[n]));
    }

    // Calculate noise level
    sigma_Rp = sigma_rel_Rp_raw * sigma_rel_Rp_scale * Rp;
    for (n in 1:N) {
        sigma_y[n] = sqrt(square(sigma_rel_y_raw * sigma_rel_y_scale * y[n]) + square(sigma_Rp[n]));
    }
}
model {
    // cov matrix for each element
    array[K] matrix[N, N] L_K;
    array[K] matrix[N, N] K_cov;

    for (k in 1:K) {
        K_cov[k] = gp_exp_quad_cov(x, alpha[k], rho[k]);
        // diagonal elements
        for (n in 1:N) {
            K_cov[k][n, n] = K_cov[k][n, n] + sq_sigma[k];
        }
        L_K[k] = cholesky_decompose(K_cov[k]);
    }

    // Cov parameter priors
    rho ~ inv_gamma(5, 5);
    alpha ~ std_normal();
    sigma ~ std_normal();

    // Gaussian process for RQ parameters
    for (k in 1:K) {
        lnR_raw[k] ~ multi_normal_cholesky(mu[k], L_K[k]);
        lntau_raw[k] ~ multi_normal_cholesky(mu[k], L_K[k]);
        beta_trans_raw[k] ~ multi_normal_cholesky(mu[k], L_K[k]);
    }

    // Relaxation surface
    for (n in 1:N) {
        y[n] ~ normal(y_hat[n], sigma_y[n]);
    }
    //Rp ~ normal(Rp_hat, sigma_Rp);

    // ZARC priors
    lnR_scale ~ inv_gamma(1, 1);
    lntau_scale ~ inv_gamma(1, 1);
    beta_trans_scale ~ inv_gamma(1, 1);
    beta_trans_mu ~ normal(0, 10);
    lnR_mu ~ normal(0, 100);
    lntau_mu ~ normal(0, 100);

    // Relative noise level prior
    sigma_rel_y_raw ~ inv_gamma(1, 1);
    sigma_rel_Rp_raw ~ inv_gamma(1, 1);
}
generated quantities {
    array[K] vector[N] lnR_raw_gp;
    array[K] vector[N] R_gp;

    for (k in 1:K) {
        lnR_raw_gp[k] = gp_pred_rng(x, lnR_raw[k], x, alpha[k], rho[k], sq_sigma[k], 1e-9);
        R_gp[k] = raw_to_actual(lnR_raw_gp[k], lnR_mu[k], lnR_scale[k]);
        R_gp[k] = exp(R_gp[k]) * R_sign[k];
    }
}
