functions {
  real map_lpdf(real a,
                vector w,
                vector mu,
                vector sigma) {
    int K = num_elements(w);
    vector[K] lps;
    for (k in 1:K)
      lps[k] = log(w[k]) + normal_lpdf(a | mu[k], sigma[k]);
    return log_sum_exp(lps);
  }
}

data {
  int<lower=1> N;              // current trial sample size
  vector[N] y;                 // outcome
  vector[N] x;                 // centered covariate
  int<lower=0, upper=1> z[N];  // treatment indicator

  // MAP prior (mixture approximation)
  int<lower=1> K;
  vector[K] w_map;
  vector[K] mu_map;
  vector<lower=0>[K] sigma_map;

  real<lower=0, upper=1> w;    // robustness weight
}

parameters {
  real alpha;                  // placebo intercept (borrowed)
  real delta;                  // treatment effect
  real beta;                   // covariate effect
  real<lower=0> sigma;         // residual SD
}

model {
  // Robust MAP prior on intercept
  target += log_mix(
    w,
    map_lpdf(alpha | w_map, mu_map, sigma_map),
    normal_lpdf(alpha | 0, 10)
  );

  // Priors for non-borrowed parameters
  delta ~ normal(0, 10);
  beta  ~ normal(0, 5);
  sigma ~ normal(0, 5);

  // Likelihood
  y ~ normal(alpha + delta * z + beta * x, sigma);
}
