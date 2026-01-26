data {
  int<lower=1> N;           // Number of patients in current trial
  vector[N] y;              // Continuous outcome
  vector[N] trt;            // Treatment indicator (0 = Placebo, 1 = Treatment)
  vector[N] x;              // Centered baseline covariate

  // Robust MAP Prior parameters (Components 1 to K-1 are MAP, K is the Vague component)
  int<lower=1> K;           
  vector<lower=0, upper=1>[K] weights; 
  vector[K] mu;
  vector[K] sigma_prior;    // Standard deviations of the mixture components
}

parameters {
  real alpha;               // Intercept (Mean of Placebo group at x=0)
  real beta;                // Treatment effect (difference)
  real gamma;               // Covariate effect
  real<lower=0> sigma_y;    // Residual standard deviation
}

model {
  // 1. Mixture Prior for alpha (Placebo Mean)
  vector[K] lps;
  for (k in 1:K) {
    lps[k] = log(weights[k]) + normal_lpdf(alpha | mu[k], sigma_prior[k]);
  }
  target += log_sum_exp(lps);

  // 2. Weakly Informative Priors for other parameters
  beta ~ normal(0, 5);       // Adjust scale based on your outcome range
  gamma ~ normal(0, 5);      
  sigma_y ~ cauchy(0, 2.5);

  // 3. Likelihood
  // Mean = alpha (placebo) + beta (if trt=1) + gamma (covariate effect)
  y ~ normal(alpha + beta * trt + gamma * x, sigma_y);
}

generated quantities {
  // You can calculate the predicted mean for each group here if desired
  real mu_placebo = alpha;
  real mu_treatment = alpha + beta;
}

