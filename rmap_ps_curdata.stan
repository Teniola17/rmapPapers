data {
  int<lower=1> N;
  vector[N] y;
  vector[N] x;

  // Intercept RMAP Components 
  int<lower=1> K_alpha;            // 
  vector[K_alpha] alphar_w;        // Weights: 
  vector[K_alpha] alphar_mu;       // Means of MAP components + robust mean
  vector[K_alpha] alphar_sd;       // SDs of MAP components + robust SD

  // Covariate RMAP Components 
  int<lower=1> K_beta;             
  vector[K_beta] betar_w;          // Weights
  vector[K_beta] betar_mu;         // Means of MAP components + robust mean
  vector[K_beta] betar_sd;         // SDs of MAP components + robust SD
}

parameters {
  real alpha;             // Current trial intercept
  real beta;              // Current trial covariate effect
  real<lower=0> sigma;    // Residual error
}

model {
  // 1. Likelihood for the Current Trial
  y ~ normal(alpha + beta * x, sigma);
  
  // Prior for Residual Sigma 
  sigma ~ exponential(1);

  // 2. RMAP Prior for Intercept (alpha)
  
  vector[K_alpha] lp_alpha;
  for (k in 1:K_alpha) {
    lp_alpha[k] = log(alphar_w[k]) + normal_lpdf(alpha | alphar_mu[k], alphar_sd[k]);
  }
  target += log_sum_exp(lp_alpha);

  // 3. RMAP Prior for Covariate (beta)
  vector[K_beta] lp_beta;
  for (k in 1:K_beta) {
    lp_beta[k] = log(betar_w[k]) + normal_lpdf(beta | betar_mu[k], betar_sd[k]);
  }
  target += log_sum_exp(lp_beta);
}