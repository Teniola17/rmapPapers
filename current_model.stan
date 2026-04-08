data {
  int<lower=1> N;                    // total sample size
  vector[N] y;                       // outcomes
  array[N] int<lower=0, upper=1> z;  // treatment indicator

  int<lower=1> M;                    // number of mixture components
  simplex[M] w_mix;                  // mixture weights
  vector[M] mu_mix;                  // mixture means
  vector<lower=0>[M] sd_mix;         // mixture SDs
}

parameters {
  real theta_star;                   // current control mean
  real beta;                         // treatment effect
  real<lower=0> sigma;               // residual SD
}

model {
  vector[M] log_comp;

  // Mixture prior on theta_star
  for (m in 1:M) {
    log_comp[m] = log(w_mix[m]) + normal_lpdf(theta_star | mu_mix[m], sd_mix[m]);
  }
  target += log_sum_exp(log_comp);

  // Weakly informative priors
  beta  ~ normal(0, 10);
  sigma ~ normal(0, 5);   // half-normal due to lower bound

  // Likelihood
  for (i in 1:N) {
    y[i] ~ normal(theta_star + beta * z[i], sigma);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  vector[M] post_mix_prob;
  vector[M] log_comp_post;

  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | theta_star + beta * z[i], sigma);
    y_rep[i]   = normal_rng(theta_star + beta * z[i], sigma);
  }

  for (m in 1:M) {
    log_comp_post[m] = log(w_mix[m]) + normal_lpdf(theta_star | mu_mix[m], sd_mix[m]);
  }

  post_mix_prob = softmax(log_comp_post);
}

