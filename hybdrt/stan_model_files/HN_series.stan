functions {
	real hn_x(real omega, real tau0, real beta) {
		// Intermediate "x" variable
		return 1 + pow(omega * tau0, beta) * cos(beta * pi() / 2);
	}
	real hn_y(real omega, real tau0, real beta){
		// Intermediate "y" variable
		return pow(omega * tau0, beta) * sin(beta * pi() / 2);
	}
	real z_mod_hn(real alpha, real x, real y) {
		return pow(sqrt(square(x) + square(y)), -alpha);
	}
	real theta_hn(real alpha, real x, real y) {
		return -alpha * atan2(y, x);
	}
}
data {
	// dimensions
	int<lower=0> N; // number of measured frequencies
	int<lower=0> K; // number of hn basis functions

	// impedance data
	vector[2 * N] z; // stacked impedance vector ([z' z'']^T)
	vector[2 * N] sigma_z; // impedance error scale estimate
	vector[N] freq; //measured frequencies

	// hn limits
	real<lower=0> min_tau_hn;
	real<lower=0> max_tau_hn;

	// fixed hyperparameters
	real<lower=0> induc_scale; // inductance scale
	real<lower=0> r_inf_scale; // ohmic resistance scale
	real<lower=0> r_hn_scale; // HN resistance scale
	real<lower=0> sigma_min_lambda;
}
transformed data {
	vector[N] omega = freq * 2 * pi();
	real min_lntau_hn = log(min_tau_hn);
	real max_lntau_hn = log(max_tau_hn);
}
parameters {
	// impedance offsets
	real<lower=0> r_inf_raw;
	real<lower=0> induc_raw;

	// hn parameters
	vector<lower=0>[K] r_hn_raw;
	vector<lower=min_lntau_hn, upper=max_lntau_hn>[K] lntau_hn;
	//ordered[K] lntau_hn;
	vector<lower=0,upper=1>[K] alpha_hn;
	vector<lower=0,upper=1>[K] beta_hn;

	real<lower=0> sigma_min;
}
transformed parameters {
	// impedance offsets
	real<lower=0> r_inf = r_inf_raw * r_inf_scale;
	real<lower=0> induc = induc_raw * induc_scale;

	// Scaled HN resistance
	vector<lower=0>[K] r_hn = r_hn_raw * r_hn_scale;

	// impedance vectors
	vector[2 * N] z_hat;
	vector[N] z_hat_re = rep_vector(0,N);
	vector[N] z_hat_im = rep_vector(0,N);

	// calculate z_hat
	for (k in 1:K){
		real tau_hn = exp(lntau_hn[k]);
		for (n in 1:N){
			real x = hn_x(omega[n], tau_hn, beta_hn[k]);
			real y = hn_y(omega[n], tau_hn, beta_hn[k]);
			real z_mod = z_mod_hn(alpha_hn[k], x, y);
			real theta = theta_hn(alpha_hn[k], x, y);
			z_hat_re[n] += r_hn[k] * z_mod * cos(theta);
			z_hat_im[n] += r_hn[k] * z_mod * sin(theta);
		}
	}

	z_hat_re += r_inf;
	z_hat_im += induc * omega;
	z_hat = append_row(z_hat_re, z_hat_im);
}
model {
	// impedance offsets
	r_inf_raw ~ std_normal();
	induc_raw ~ std_normal();

	// hn parameters
	r_hn_raw ~ std_normal();

	// impedance
	z ~ normal(z_hat, sigma_z + sigma_min);
	sigma_min ~ exponential(sigma_min_lambda);
}