
data {
  int<lower=1> H;
  vector[H] ybar;
  vector<lower=0>[H] se;
}
parameters {
  vector[H] theta;
  real mu;
  real<lower=0> tau;
}
model {
  mu ~ normal(0, 10);
  tau ~ normal(0, 5);
  theta ~ normal(mu, tau);
  ybar ~ normal(theta, se);
}
generated quantities {
  real theta_star;
  theta_star = normal_rng(mu, tau);
}

