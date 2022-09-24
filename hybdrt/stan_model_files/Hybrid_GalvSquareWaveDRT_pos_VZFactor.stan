data {
	// dimensions
	int<lower=0> N_t; // number of measured times
	int<lower=0> N_f; // number of measured frequencies
	int<lower=0> K; // number of basis functions
	
	// voltage response data
	vector[N_t] V; // voltage vector

	// impedance data
	vector[2 * N_f] Z;  // impedance vector (concatenated)
	
	// response matrices
	matrix[N_t, K] A_t; // time response matrix
	matrix[2 * N_f, K] A_f; // impedance matrix

	// Response vectors
	vector[N_t] irv; // inductance time response
	vector[N_t] inf_rv; // R_inf time response
	vector[2 * N_f] izv; // inductance impedance vector
	vector[2 * N_f] inf_zv; // R_inf impedance vector

	// Penalty matrices
	matrix[K,K] L0; // 0th order differentiation matrix
	matrix[K,K] L1; // 1st order differentiation matrix
	matrix[K,K] L2; // 2nd order differentiation matrix
	
	// fixed hyperparameters
	real<lower=0> ups_alpha; // shape for inverse gamma distribution on ups
	real<lower=0> ups_beta; // rate for inverse gamma distribution on ups
	real<lower=0> ups_scale;
	real<lower=0> ds_alpha;
	real<lower=0> ds_beta;
	real<lower=0> v_sigma_min; // noise level floor (time response)
	real<lower=0> v_sigma_res_scale;
	real<lower=0> z_sigma_min; // noise level floor (impedance)
	real<lower=0> z_sigma_scale;
	real<lower=0> z_alpha_scale;

	real<lower=0> inductance_scale;
	real<lower=0> R_inf_scale;
	real<lower=0> V_baseline_scale;
	real<lower=0> log_VZ_factor_scale;
}
parameters {
    // Offsets
	real<lower=0> R_inf_raw;
	real V_baseline_raw;
	real<lower=0> inductance_raw;

	// Z factor
	real log_VZ_factor_raw;

	// DRT coefficients
	vector<lower=0>[K] x;

	// time response error structure
	real<lower=0> v_sigma_res_raw;

	// impedance error structure
    real<lower=0> z_sigma_res_raw;
	real<lower=0> z_alpha_prop_raw;
	real<lower=0> z_alpha_re_raw;
	real<lower=0> z_alpha_im_raw;

	vector<lower=0>[K] ups_raw;
	real<lower=0> d0_strength;
	real<lower=0> d1_strength;
	real<lower=0> d2_strength;
}
transformed parameters {
	// offsets
	real<lower=0> R_inf = R_inf_raw * R_inf_scale;
	real V_baseline = V_baseline_raw*V_baseline_scale;
	real<lower=0> inductance = inductance_raw * inductance_scale;
	vector[N_t] V_inst = R_inf * inf_rv;  // instantaneous voltage response

	// VZ_factor
	real<lower=0> VZ_factor = exp(log_VZ_factor_raw * log_VZ_factor_scale);
	
	// calculated voltage
	vector[N_t] V_hat = (A_t * x + V_inst + V_baseline) * VZ_factor;
	
	// time response error structure
	real<lower=0> v_sigma_res = v_sigma_res_raw * v_sigma_res_scale;
	real<lower=0> v_sigma_tot = sqrt(square(v_sigma_min) + square(v_sigma_res));

	// impedance error structure
	real<lower=0> z_sigma_res = z_sigma_res_raw * z_sigma_scale;
	real<lower=0> z_alpha_prop = z_alpha_prop_raw * z_alpha_scale;
	real<lower=0> z_alpha_re = z_alpha_re_raw * z_alpha_scale;
	real<lower=0> z_alpha_im = z_alpha_im_raw * z_alpha_scale;
	vector[2 * N_f] Z_hat = (A_f * x + R_inf * inf_zv + inductance * izv) / VZ_factor;
	vector[N_f] Z_hat_re = Z_hat[1: N_f];
	vector[N_f] Z_hat_im = Z_hat[N_f + 1: 2 * N_f];
	vector<lower=0>[2 * N_f] z_sigma_tot = sqrt(
	    square(z_sigma_min) + square(z_sigma_res) + square(z_alpha_prop * Z_hat)
		+ square(z_alpha_re * append_row(Z_hat_re, Z_hat_re))
		+ square(z_alpha_im * append_row(Z_hat_im, Z_hat_im))
    );
	
	// complexity
	vector<lower=0>[K] q;
	vector<lower=0>[K] ups = ups_raw * ups_scale;
	vector[K-2] dups;
	
	// calculate complexity and upsilon
	q = sqrt(d0_strength*square(L0*x) + d1_strength*square(L1*x) + d2_strength*square(L2*x));
	for (k in 1:K-2)
		dups[k] = 0.5*(ups[k+1] - 0.5*(ups[k] + ups[k+2]))/ups[k+1];
}
model {
    // Penalty strengths
	d0_strength ~ inv_gamma(ds_alpha, ds_beta);
	d1_strength ~ inv_gamma(ds_alpha, ds_beta);
	d2_strength ~ inv_gamma(ds_alpha, ds_beta);

	// Offsets
	R_inf_raw ~ std_normal();
	V_baseline_raw ~ std_normal();
	inductance_raw ~ std_normal();

	// VZ factor
	log_VZ_factor_raw ~ std_normal();

    // DRT complexity
	q ~ normal(0, ups);
	ups_raw ~ inv_gamma(ups_alpha, ups_beta);
	dups ~ std_normal();

	// time response error structure
	V ~ normal(V_hat, v_sigma_tot);
	v_sigma_res_raw ~ std_normal();

	// impedance error structure
	Z ~ normal(Z_hat, z_sigma_tot);
	z_sigma_res_raw ~ std_normal();
	z_alpha_prop_raw ~ std_normal();
	z_alpha_re_raw ~ std_normal();
	z_alpha_im_raw ~ std_normal();
}