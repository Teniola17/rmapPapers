data {
  int<lower=1> H;                 // number of historical studies
  vector[H] ybar;                 // observed historical means
  vector<lower=0>[H] se;          // standard errors of historical means
}

parameters {
  vector[H] theta;                // latent historical study means
  real mu;                        // overall mean
  real<lower=0> tau;              // between-study SD
}

model {
  // Hyperpriors
  mu ~ normal(0, 10);
  tau ~ normal(0, 5);             // half-normal because tau > 0

  // Exchangeable MAP prior
  theta ~ normal(mu, tau);

  // Historical likelihood
  ybar ~ normal(theta, se);
}

generated quantities {
  vector[H] log_lik;
  real theta_star;                // predictive current control mean

  for (h in 1:H) {
    log_lik[h] = normal_lpdf(ybar[h] | theta[h], se[h]);
  }

  // MAP predictive prior for current control
  theta_star = normal_rng(mu, tau);
}

