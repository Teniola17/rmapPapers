
data {
  int<lower=1> H;
  vector[H] ybar;
  vector<lower=0>[H] se;
  vector[H] time;
  real<lower=time[H]> t_star;
}
transformed data {
  vector[H - 1] delta;
  real<lower=0> delta_star;
  for (h in 2:H) {
    delta[h - 1] = time[h] - time[h - 1];
  }
  delta_star = t_star - time[H];
}
parameters {
  vector[H] theta;
  real mu;
  real mu1;
  real<lower=0> sigma1;
  real<lower=0, upper=1> rho;
  real<lower=0> tau;
}
model {
  mu ~ normal(0, 10);
  mu1 ~ normal(mu, 5);
  sigma1 ~ normal(0, 5);
  rho ~ beta(2, 2);
  tau ~ normal(0, 5);

  theta[1] ~ normal(mu1, sigma1);

  for (h in 2:H) {
    real mean_h;
    real var_h;
    mean_h = mu + pow(rho, delta[h - 1]) * (theta[h - 1] - mu);
    var_h  = square(tau) * (1 - pow(rho, 2 * delta[h - 1])) / (1 - square(rho));
    theta[h] ~ normal(mean_h, sqrt(var_h));
  }

  ybar ~ normal(theta, se);
}
generated quantities {
  real mean_star;
  real var_star;
  real theta_star;

  mean_star = mu + pow(rho, delta_star) * (theta[H] - mu);
  var_star  = square(tau) * (1 - pow(rho, 2 * delta_star)) / (1 - square(rho));
  theta_star = normal_rng(mean_star, sqrt(var_star));
}

