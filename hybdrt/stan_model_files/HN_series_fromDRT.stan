functions {
	real theta_hn(real t, real beta) {
		return atan2(sin(pi() * beta), pow(t, beta) + cos(pi() * beta));
	}
	real gamma_hn(real t, real alpha, real beta) {
	    real theta = theta_hn(t, beta);
	    real nume = pow(t, beta * alpha) * sin(alpha * theta);
	    real deno = 2 * pi() * pow(1 + 2 * cos(pi() * beta) * pow(t, beta) + pow(t, 2 * beta), alpha / 2);
	    return nume / deno;
	}
}
data {
	// dimensions
	int<lower=0> M; // number of DRT evaluation points
	int<lower=0> K; // number of hn basis functions

	// DRT input
	vector[M] tau; // tau
	vector[M] gamma; // estimated DRT

	// hn limits
	real<lower=0> min_tau_hn;
	real<lower=0> max_tau_hn;

	// fixed hyperparameters
	real<lower=0> r_hn_scale; // HN resistance scale
	real<lower=0> sigma_alpha_scale;
	real<lower=0> sigma_const_scale;
}
transformed data {
	real min_lntau_hn = log(min_tau_hn);
	real max_lntau_hn = log(max_tau_hn);
}
parameters {
	// hn parameters
	vector<lower=0>[K] r_hn_raw;
	vector<lower=min_lntau_hn, upper=max_lntau_hn>[K] lntau_hn;
	//ordered[K] lntau_hn;
	vector<lower=0,upper=1>[K] alpha_hn;
	vector<lower=0,upper=1>[K] beta_hn;

    // error structure
	real<lower=0> sigma_alpha_raw;
	real<lower=0> sigma_const_raw;
}
transformed parameters {
	// Scaled HN resistance
	vector<lower=0>[K] r_hn = r_hn_raw * r_hn_scale;

	// gamma vectors
	real<lower=0> sigma_alpha = sigma_alpha_raw * sigma_alpha_scale;
	real<lower=0> sigma_const = sigma_const_raw * sigma_const_scale;
	vector<lower=0>[M] sigma_gamma = sigma_alpha * gamma + sigma_const;
	vector[M] gamma_hat = rep_vector(0, M);

	// calculate gamma_hat
	for (k in 1:K){
		real tau_hn = exp(lntau_hn[k]);
		for (m in 1:M){
			real t = tau[m] / tau_hn;
			gamma_hat[m] += r_hn[k] * gamma_hn(t, alpha_hn[k], beta_hn[k]);
		}
	}
}
model {
	// hn parameters
	r_hn_raw ~ std_normal();

	// gamma
	gamma ~ normal(gamma_hat, sigma_gamma);
	sigma_alpha_raw ~ std_normal();
	sigma_const_raw ~ std_normal();
}