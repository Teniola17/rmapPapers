data {
  int<lower=1> N;
  vector[N] y;
  vector[N] trt;
  vector[N] x;

  int<lower=1> K;
  vector[K] weights;
  vector[K] mu;
  vector[K] sigma;
}

parameters {
  real alpha;
  real beta;
  real gamma;
  real<lower=0> sigma_y;
}

model {
  vector[K] lps;

  // MAP prior for intercept (placebo mean)
  for (k in 1:K) {
    lps[k] = log(weights[k]) +
             normal_lpdf(alpha | mu[k], sigma[k]);
  }
  target += log_sum_exp(lps);

  // Weak priors for new information
  beta ~ normal(0, 5);
  gamma ~ normal(0, 5);
  sigma_y ~ cauchy(0, 2.5);

  // Likelihood
  y ~ normal(alpha + beta .* trt + gamma .* x,
             sigma_y);
}
