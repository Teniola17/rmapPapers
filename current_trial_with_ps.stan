data {
  int<lower=1> N;
  vector[N] y;
  vector[N] trt;
  vector[N] x;

  // Intercept MAP
  int<lower=1> K_alpha;
  vector[K_alpha] w_alpha;
  vector[K_alpha] mu_alpha;
  vector[K_alpha] sigma_alpha;

  // Covariate MAP
  int<lower=1> K_gamma;
  vector[K_gamma] w_gamma;
  vector[K_gamma] mu_gamma;
  vector[K_gamma] sigma_gamma;
}

parameters {
  real alpha;
  real beta;
  real gamma;
  real<lower=0> sigma_y;
}

model {
  vector[K_alpha] lp_alpha;
  vector[K_gamma] lp_gamma;

  // MAP prior for intercept
  for (k in 1:K_alpha)
    lp_alpha[k] = log(w_alpha[k]) +
                  normal_lpdf(alpha | mu_alpha[k], sigma_alpha[k]);
  target += log_sum_exp(lp_alpha);

  // MAP prior for covariate effect
  for (k in 1:K_gamma)
    lp_gamma[k] = log(w_gamma[k]) +
                  normal_lpdf(gamma | mu_gamma[k], sigma_gamma[k]);
  target += log_sum_exp(lp_gamma);

  // Weak prior for treatment
  beta ~ normal(0, 5);
  sigma_y ~ cauchy(0, 2.5);

  // Likelihood
  y ~ normal(alpha + beta .* trt + gamma .* x, sigma_y);
}



