data {
    int<lower=1> N;
    int<lower=1> Dx;
    int<lower=1> Dy;
    array[N] vector[Dx] x;
    array[N] vector[Dy] y;
}
transformed_data {
    array[N] vector[Dy] mu;
    for (n in 1:N) {
        mu[n] = rep_vector(0, Dy);
    }
}
parameters {
    vector<lower=0>[Dx] rho;  // length scale
    vector<lower=0>[Dy] alpha; // magnitude
    vector<lower=0>[Dy]
}