data {
  int<lower=1> N;                 // total number of observations
  int<lower=1> J;                 // number of historical trials
  array[N] int<lower=1, upper=J> trial;  // trial indicator
  vector[N] y;                    // outcome
  vector[N] x;                    // covariate
}

parameters {
  real mu_alpha;                  // overall placebo intercept
  real<lower=0> tau_alpha;        // between-trial SD
  vector[J] alpha_raw;            // non-centered intercepts
  real gamma;                      // covariate effect
  real<lower=0> sigma;            // residual SD
}

transformed parameters {
  vector[J] alpha;
  alpha = mu_alpha + tau_alpha * alpha_raw;
}

model {
  // Priors
  mu_alpha  ~ normal(0, 10);
  tau_alpha ~ normal(0, 2);       // half-normal via <lower=0>
  alpha_raw ~ normal(0, 1);

  gamma  ~ normal(0, 5);
  sigma ~ normal(0, 5);

  // Likelihood (historical placebo only)
  for (n in 1:N) {
    y[n] ~ normal(alpha[trial[n]] + gamma * x[n], sigma);
  }
}

generated quantities {
  real alpha_new;                 // MAP prior draw for new trial intercept
  alpha_new = normal_rng(mu_alpha, tau_alpha);
}
