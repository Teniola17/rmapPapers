data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1,upper=J> trial[N];
  vector[N] y;
  vector[N] x;   // centered
}

parameters {
  vector[2] mu;                     // (mu_alpha, mu_gamma)
  vector<lower=0>[2] tau;           // SDs
  cholesky_factor_corr[2] L_Omega;  // correlation

  matrix[2, J] theta;               // [1]=alpha_j, [2]=gamma_j
  real<lower=0> sigma;
}

transformed parameters {
  matrix[2, J] theta_tilde;
  theta_tilde = diag_pre_multiply(tau, L_Omega) * theta;
}

model {
  // Hyperpriors
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 2.5);
  L_Omega ~ lkj_corr_cholesky(2);
  sigma ~ cauchy(0, 2.5);

  // Non-centered parameterization
  to_vector(theta) ~ normal(0, 1);

  // Likelihood
  for (n in 1:N) {
    y[n] ~ normal(
      mu[1] + theta_tilde[1, trial[n]]
      + (mu[2] + theta_tilde[2, trial[n]]) * x[n],
      sigma
    );
  }
}

generated quantities {
  vector[2] theta_new;
  theta_new = multi_normal_rng(
    mu,
    quad_form_diag(
      multiply_lower_tri_self_transpose(L_Omega),
      tau
    )
  );
}



