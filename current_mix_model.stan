
data {
  int<lower=1> N;
  vector[N] y;
  array[N] int<lower=0, upper=1> z;

  int<lower=1> M;
  simplex[M] w_mix;
  vector[M] mu_mix;
  vector<lower=0>[M] sd_mix;
}
parameters {
  real theta_star;
  real beta;
  real<lower=0> sigma;
}
model {
  vector[M] log_comp;

  for (m in 1:M) {
    log_comp[m] = log(w_mix[m]) + normal_lpdf(theta_star | mu_mix[m], sd_mix[m]);
  }
  target += log_sum_exp(log_comp);

  beta ~ normal(0, 10);
  sigma ~ normal(0, 5);

  for (i in 1:N) {
    y[i] ~ normal(theta_star + beta * z[i], sigma);
  }
}
generated quantities {
  real p_benefit;
  p_benefit = Phi((-beta) / 1e-9); // placeholder not useful; will use draws in R
}

