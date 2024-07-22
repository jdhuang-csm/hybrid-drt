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
    vector ZARC_Z(vector freq, real tau_0, real beta) {
        int M = size(freq);
        real pb2 = pi() * beta / 2; 
        vector[2 * M] z_out;
        for (m in 1:M) {
            real wt = pow(freq[m] * 2 * pi() * tau_0, beta);
            real a = 1 + wt * cos(pb2); // real part
            real b = wt * sin(pb2); // imag part
            real deno = square(a) + square(b);

            z_out[m] = a / deno;
            z_out[M + m] = -b / deno;
        }
        return z_out;
    }
	vector raw_to_actual(vector x_raw, real mu, real scale) {
	    return mu + x_raw * scale;
	}
	vector gp_pred_rng(array[] vector x2,
                     vector y1,
                     array[] vector x1,
                     real alpha,
                     array[] real rho,
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
    int<lower=1> N; // number of measurements
    int<lower=1> D; // x dimensions
    int<lower=1> M; // tau grid size
    int<lower=1> K; // number of discrete elements
    vector<lower=0>[M] f; // frequencies
    array[N] vector[D] x; // data matrix
    array[N] vector[2 * M] z; // impedance
    array[N] vector[2 * M] z_weight; // impedance weights
    real<lower=0> sigma_gp_scale;
    real<lower=0> sigma_zmod_scale;
    real<lower=0> sigma_zuniform_scale;
    vector[K] R_sign;
}
transformed data {
    array[K] vector[N] mu; // GP mean
    array[2] vector[N] mu_offsets; // GP mean for offsets
    array[N] vector[M] zmod; // Impedance modulus

    for (i in 1:2) {
        mu_offsets[i] = rep_vector(0.0, N);
    }
    for (k in 1:K) {
        mu[k] = rep_vector(0.0, N);
    }
    for (n in 1:N) {
        zmod[n] = sqrt(square(z[n][:M]) + square(z[n][M + 1:]));
    }
}
parameters {
    array[K, D] real<lower=0> rho;  // cov length scale
    vector<lower=0>[K] alpha; // discrete element cov magnitude
    vector<lower=0>[K] sigma; // discrete element noise level

    // offsets
    vector[N] lnOhmic_raw;
    real<lower=0> lnOhmic_scale;
    real lnOhmic_mu;
    vector[N] lnInduc_raw;
    real<lower=0> lnInduc_scale;
    real lnInduc_mu;
    // GP hypers for offsets
    array[2, D] real<lower=0> rho_offsets;
    vector<lower=0>[2] alpha_offsets; // offset cov magnitude
    vector<lower=0>[2] sigma_offsets; // offset noise level
    

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
    real<lower=0> sigma_zmod_raw; // modulus noise level
    real<lower=0> sigma_zuniform_raw; // uniform noise level
}
transformed parameters {
    vector[N] lnOhmic;
    vector[N] Ohmic;
    vector[N] lnInduc;
    vector[N] Induc;

    array[K] vector[N] lnR;
    array[K] vector[N] R;
    array[K] vector[N] lntau;
    array[K] vector[N] beta_trans;
    array[K] vector<lower=0, upper=1>[N] beta;
    array[N] vector<lower=0>[2 * M] sigma_z;
    real<lower=0> sigma_zuniform = sigma_zuniform_raw * sigma_zuniform_scale; // uniform contribution
    vector<lower=0>[K] sq_sigma = square(sigma_gp_scale * sigma);
    vector<lower=0>[2] sq_sigma_offsets = square(sigma_gp_scale * sigma_offsets);

    // model surface
    array[N] vector[2 * M] z_hat;

    // Transform raw offsets
    lnOhmic = raw_to_actual(lnOhmic_raw, lnOhmic_mu, lnOhmic_scale);
    lnInduc = raw_to_actual(lnInduc_raw, lnInduc_mu, lnInduc_scale);
    Ohmic = exp(lnOhmic);
    Induc = exp(lnInduc);

    // Transform raw RQ parameters to actual values
    for (k in 1:K) {
        lnR[k] = raw_to_actual(lnR_raw[k], lnR_mu[k], lnR_scale[k]);
        R[k] = exp(lnR[k]) * R_sign[k];
        lntau[k] = raw_to_actual(lntau_raw[k], lntau_mu[k], lntau_scale[k]);
        beta_trans[k] = raw_to_actual(beta_trans_raw[k], beta_trans_mu[k], beta_trans_scale[k]);
        beta[k] = exp(beta_trans[k]) ./ (exp(beta_trans[k]) + 1.0);
    }

    // calculate model impedance surface
    for (n in 1:N) {
        // offset contributions
        //z_hat[n] = rep_vector(0.0, 2 * M);
        z_hat[n] = append_row(rep_vector(Ohmic[n], M), 2 * pi() * f * Induc[n]);
        // ZARC contributions
        for (k in 1:K) {
            z_hat[n] = z_hat[n] + R[k][n] * ZARC_Z(f, exp(lntau[k][n]), beta[k][n]);
        }
    }

    // Calculate noise level
    for (n in 1:N) {
        sigma_z[n] = sqrt(
            square(sigma_zmod_raw * sigma_zmod_scale * append_row(zmod[n], zmod[n])) 
            + square(sigma_zuniform)
            );
        // apply user-provided weights
        sigma_z[n] = sigma_z[n] ./ z_weight[n];
    }
}
model {
    // GP model for offsets
    // ----------------------------
    // cov matrices for offsets
    array[2] matrix[N, N] L_K_offsets;
    array[2] matrix[N, N] K_cov_offsets;

    for (i in 1:2) {
        K_cov_offsets[i] = gp_exp_quad_cov(x, alpha_offsets[i], rho_offsets[i]);
        // diagonal elements
        for (n in 1:N) {
            K_cov_offsets[i][n, n] = K_cov_offsets[i][n, n] + sq_sigma_offsets[i];
        }
        L_K_offsets[i] = cholesky_decompose(K_cov_offsets[i]);
    }

    // Offset cov parameter priors
    for (i in 1:2) {
        rho_offsets[i] ~ inv_gamma(5, 5);
    }
    alpha_offsets ~ std_normal();
    sigma_offsets ~ std_normal();

    // Gaussian process prior
    lnOhmic_raw ~ multi_normal_cholesky(mu_offsets[1], L_K_offsets[1]);
    lnInduc_raw ~ multi_normal_cholesky(mu_offsets[2], L_K_offsets[2]);
    
    // GP model for ZARC elements
    // ----------------------------
    // cov matrices for ZARC elements
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
    for (k in 1:K) {
        rho[k] ~ inv_gamma(5, 5);
    }
    alpha ~ std_normal();
    sigma ~ std_normal();

    // Gaussian process for RQ parameters
    for (k in 1:K) {
        lnR_raw[k] ~ multi_normal_cholesky(mu[k], L_K[k]);
        lntau_raw[k] ~ multi_normal_cholesky(mu[k], L_K[k]);
        beta_trans_raw[k] ~ multi_normal_cholesky(mu[k], L_K[k]);
    }

    // Impedance surface
    for (n in 1:N) {
        z[n] ~ normal(z_hat[n], sigma_z[n]);
    }

    // ZARC priors
    lnR_scale ~ inv_gamma(1, 1);
    lntau_scale ~ inv_gamma(1, 1);
    beta_trans_scale ~ inv_gamma(1, 1);
    beta_trans_mu ~ normal(0, 10);
    lnR_mu ~ normal(0, 100);
    lntau_mu ~ normal(0, 100);

    // Relative noise level prior
    sigma_zmod_raw ~ inv_gamma(1, 1);
    sigma_zuniform_raw ~ inv_gamma(1, 1);
}
generated quantities {
    vector[N] lnOhmic_raw_gp;
    vector[N] Ohmic_gp;
    vector[N] lnInduc_raw_gp;
    vector[N] Induc_gp;
    array[K] vector[N] lnR_raw_gp;
    array[K] vector[N] R_gp;

    lnOhmic_raw_gp = gp_pred_rng(x, lnOhmic_raw, x, alpha_offsets[1], 
        rho_offsets[1], sq_sigma_offsets[1], 1e-9);
    Ohmic_gp = raw_to_actual(lnOhmic_raw_gp, lnOhmic_mu, lnOhmic_scale);
    Ohmic_gp = exp(Ohmic_gp);

    lnInduc_raw_gp = gp_pred_rng(x, lnInduc_raw, x, alpha_offsets[2], 
        rho_offsets[2], sq_sigma_offsets[2], 1e-9);
    Induc_gp = raw_to_actual(lnInduc_raw_gp, lnInduc_mu, lnInduc_scale);
    Induc_gp = exp(Induc_gp);

    for (k in 1:K) {
        lnR_raw_gp[k] = gp_pred_rng(x, lnR_raw[k], x, alpha[k], rho[k], sq_sigma[k], 1e-9);
        R_gp[k] = raw_to_actual(lnR_raw_gp[k], lnR_mu[k], lnR_scale[k]);
        R_gp[k] = exp(R_gp[k]) * R_sign[k];
    }
}
