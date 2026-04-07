library(cmdstanr)


stan_code <- 'data {
  int<lower=1> H;                    // number of historical studies
  vector[H] ybar;                    // historical study means
  vector<lower=0>[H] se;             // standard errors of study means
  vector[H] time;                    // ordered study times
  real<lower=time[H]> t_star;        // current trial time
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
  vector[H] theta;                   // latent historical control means

  real mu;                           // long-run mean
  real mu1;                          // initial mean
  real<lower=0> sigma1;              // initial SD
  real<lower=0, upper=1> rho;        // temporal dependence
  real<lower=0> tau;                 // innovation SD
}

model {
  // Hyperpriors
  mu ~ normal(0, 10);
  mu1 ~ normal(mu, 5);
  sigma1 ~ normal(0, 5);             // half-normal by truncation
  rho ~ beta(2, 2);
  tau ~ normal(0, 5);                // half-normal by truncation

  // Initial distribution
  theta[1] ~ normal(mu1, sigma1);

  // Irregular-time AR(1) evolution
  for (h in 2:H) {
    real mean_h;
    real var_h;

    mean_h = mu + pow(rho, delta[h - 1]) * (theta[h - 1] - mu);
    var_h  = square(tau) * (1 - pow(rho, 2 * delta[h - 1])) / (1 - square(rho));

    theta[h] ~ normal(mean_h, sqrt(var_h));
  }

  // Historical likelihood
  ybar ~ normal(theta, se);
}

generated quantities {
  vector[H] log_lik;
  vector[H] xi;                      // non-stationarity adjustment
  real mean_star;
  real var_star;
  real theta_star;                   // predictive current control mean

  for (h in 1:H) {
    log_lik[h] = normal_lpdf(ybar[h] | theta[h], se[h]);
  }

  for (h in 1:H) {
    xi[h] = pow(rho, 2 * (time[h] - time[1])) *
            (square(sigma1) - square(tau) / (1 - square(rho)));
  }

  mean_star = mu + pow(rho, delta_star) * (theta[H] - mu);
  var_star  = square(tau) * (1 - pow(rho, 2 * delta_star)) / (1 - square(rho));

  theta_star = normal_rng(mean_star, sqrt(var_star));
}
'

writeLines(stan_code, con = "file.stan")

library(cmdstanr)
library(posterior)
library(bayesplot)
library(RBesT)

hist_df <- data.frame(
  study_id   = c("H1", "H2", "H3", "H4"),
  time       = c(2015, 2017, 2020, 2023),
  n_ctrl     = c(80, 95, 88, 110),
  mean_ctrl  = c(12.4, 11.9, 11.1, 10.8),
  sd_ctrl    = c(2.1, 2.3, 2.0, 1.9)
)

hist_df$se <- hist_df$sd_ctrl / sqrt(hist_df$n_ctrl)

stan_data <- list(
  H    = nrow(hist_df),
  ybar = hist_df$mean_ctrl,
  se   = hist_df$se,
  time = hist_df$time,
  t_star = 2024
)

mod <- cmdstan_model("file.stan")

fit <- mod$sample(
  data = stan_data,
  seed = 1234,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 200
)

fit$summary()

theta_star_draws <- fit$draws(variables = "theta_star", format = "df")[["theta_star"]]

# Quick look
summary(theta_star_draws)
hist(theta_star_draws, breaks = 30, main = expression(theta[star]), xlab = "theta_star")

# Fit finite normal mixture prior
mix_prior <- automixfit(theta_star_draws)

# Inspect mixture
print(mix_prior)
plot(mix_prior)

# Optional: compare empirical draws and fitted mixture density
plot(density(theta_star_draws), main = expression("Predictive draws and fitted mixture for " * theta[star]))
plot(mix_prior, add = TRUE)


