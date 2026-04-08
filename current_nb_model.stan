
data {
  int<lower=1> N;
  vector[N] y;
  array[N] int<lower=0, upper=1> z;
}
parameters {
  real theta_star;
  real beta;
  real<lower=0> sigma;
}
model {
  theta_star ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ normal(0, 5);

  for (i in 1:N) {
    y[i] ~ normal(theta_star + beta * z[i], sigma);
  }
}

