data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> trial; 
  vector[N] y;
  vector[N] x;        // centered baseline covariate
}

parameters {
  real mu_alpha;
  real<lower=0> tau_alpha;

  vector[J] alpha;
  real gamma;
  real<lower=0> sigma;
}

model {
  // Hyperpriors
  mu_alpha ~ normal(0, 10);
  tau_alpha ~ cauchy(0, 2.5);
  gamma ~ normal(0, 5);
  sigma ~ cauchy(0, 2.5);

  // Hierarchical prior
  alpha ~ normal(mu_alpha, tau_alpha);

  // Likelihood
  // FIX 2: Use '*' for scalar (gamma) times vector (x)
  // alpha[trial] now works because trial is an integer array
  y ~ normal(alpha[trial] + gamma * x, sigma);
}

generated quantities {
  real alpha_new;
  alpha_new = normal_rng(mu_alpha, tau_alpha);
}

