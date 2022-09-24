functions {
	vector unit_step (vector times, real t_step) {
		int M = rows(times);
		vector[M] u;
		for (m in 1:M) {
			if (times[m] > t_step) {
				u[m] = 1;
			}
			else {
				u[m] = 0;
			}
		}
		return u;
	}
}
data {
	// dimensions
	int<lower=0> N; // number of measured times
	int<lower=0> K; // number of basis functions
	// int<lower=0> N_steps; // number of steps
	
	// current signal info
	// vector[N_steps] t_steps; // times at which steps occurred
	// vector[N_steps] I_steps; // sign and magnitude of current steps
	// vector[N] I;
	
	// voltage response data
	vector[N] times; // times
	vector[N] V; // voltage vector
	
	// Response matrix
	matrix[N,K] A; // response matrix

	// response vectors
	vector[N] inductance_response;
	vector[N] inf_rv; // R_inf response

	// Penalty matrices
	matrix[K,K] L0; // 0th order differentiation matrix
	matrix[K,K] L1; // 1st order differentiation matrix
	matrix[K,K] L2; // 2nd order differentiation matrix
	
	// fixed hyperparameters
	real<lower=0> sigma_min; // noise level floor
	real<lower=0> ups_alpha; // shape for inverse gamma distribution on ups
	real<lower=0> ups_beta; // rate for inverse gamma distribution on ups
	real<lower=0> ups_scale;
	real<lower=0> ds_alpha;
	real<lower=0> ds_beta;
	real<lower=0> sigma_res_scale;
	// real<lower=0> induc_scale;
	real<lower=0> R_inf_scale;
	real<lower=0> inductance_scale;
	real<lower=0> V_baseline_scale;
}
transformed data{
	// current input
	// vector[N] I = rep_vector(0, N);
	// for (s in 1:N_steps){
	// 	I += I_steps[s]*unit_step(times, t_steps[s]);
	// }
}
parameters {
	real<lower=0> R_inf_raw;
	real<lower=0> inductance_raw;
	real V_baseline_raw;
	vector<lower=0>[K] x; // DRT coefficients
	real<lower=0> sigma_res_raw;
	// real<lower=0> alpha_prop_raw;
	// real<lower=0> alpha_re_raw;
	// real<lower=0> alpha_im_raw;
	vector<lower=0>[K] ups_raw;
	real<lower=0> d0_strength;
	real<lower=0> d1_strength;
	real<lower=0> d2_strength;
}
transformed parameters {
	// offsets
	real<lower=0> R_inf = R_inf_raw*R_inf_scale;
	real<lower=0> inductance = inductance_raw*inductance_scale;
	real V_baseline = V_baseline_raw*V_baseline_scale;
	vector[N] V_inst = R_inf * inf_rv;  // instantaneous voltage response
	
	// calculated voltage
	vector[N] V_hat = A*x + V_inst + V_baseline + inductance*inductance_response;
	
	// error structure
	real<lower=0> sigma_res = sigma_res_raw*sigma_res_scale;
	real<lower=0> sigma_tot = sqrt(square(sigma_min) + square(sigma_res));
	
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
	d0_strength ~ inv_gamma(ds_alpha,ds_beta); 
	d1_strength ~ inv_gamma(ds_alpha,ds_beta);
	d2_strength ~ inv_gamma(ds_alpha,ds_beta);
	ups_raw ~ inv_gamma(ups_alpha,ups_beta);
	R_inf_raw ~ std_normal();
	inductance_raw ~ std_normal();
	V_baseline_raw ~ std_normal();
	// induc_raw ~ std_normal();
	q ~ normal(0,ups);
	dups ~ std_normal();
	V ~ normal(V_hat,sigma_tot);
	sigma_res_raw ~ std_normal();
	// alpha_prop_raw ~ std_normal();
	// alpha_re_raw ~ std_normal();
	// alpha_im_raw ~ std_normal();
}