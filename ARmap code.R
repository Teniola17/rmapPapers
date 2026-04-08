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
  t_star = 2026
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


### THE CURRENT TRIAL

# simulate the current trial data

# -----------------------------
# Simulate current trial
# -----------------------------
n_ctrl <- 150
n_trt  <- 200
n      <- n_ctrl + n_trt

theta_star_true <- 10.5   # current control mean
beta_true       <- -1.2   # treatment effect
sigma_true      <- 2.0    # residual SD

z <- c(rep(0, n_ctrl), rep(1, n_trt))
y <- rnorm(
  n = n,
  mean = theta_star_true + beta_true * z,
  sd = sigma_true
)

current_df <- data.frame(
  id  = 1:n,
  arm = z,
  y   = y
)

head(current_df)
table(current_df$arm)

current_df %>% group_by(arm) %>% summarize(Mean=mean(y))



new_mod <- 'data {
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
'

writeLines(new_mod, con = "current_model.stan")

alpha_m <- mix_prior["m",]
alpha_s <- mix_prior["s",]
alpha_w <- mix_prior["w",]

stan_data_current <- list(
  N = nrow(current_df),
  y = current_df$y,
  z = current_df$arm,
  M = length(alpha_w),
  w_mix = alpha_w,
  mu_mix = alpha_m,
  sd_mix = alpha_s
)

mod_current <- cmdstan_model("current_model.stan")


fit_current <- mod_current$sample(
  data = stan_data_current,
  seed = 5678,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 200
)

fit_current$summary()

# some diagnostic plots

posterior <- as_draws_df(fit_current$draws())
mcmc_trace(posterior, pars = c("theta_star", "beta", "sigma"))

######## Robustify the AR(1) predictive prior by adding a vague component

mix_prior_r <- robustify(mix_prior, weight = 0.2, mean = 0, sigma = 1)

alpha_m_r <- mix_prior_r["m",]
alpha_s_r <- mix_prior_r["s",]
alpha_w_r <- mix_prior_r["w",]

print(mix_prior_r)
plot(mix_prior_r)

stan_data_current_r <- list(
  N = nrow(current_df),
  y = current_df$y,
  z = current_df$arm,
  M = length(alpha_w_r),
  w_mix = alpha_w_r,
  mu_mix = alpha_m_r,
  sd_mix = alpha_s_r
)

#mod_current_rmap <- cmdstan_model("current_model.stan")


fit_current_r <- mod_current$sample(
  data = stan_data_current_r,
  seed = 5678,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 200
)

fit_current_r$summary()

# map prior stan model 

MAP1 <- 'data {
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
'

writeLines(MAP1, con = "map_prior.stan")

mod_map <- cmdstan_model("map_prior.stan")

fit_map <- mod_map$sample(
  data = stan_data,
  seed = 91011,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 200
)

fit_map$summary()

theta_draws_map <- fit_map$draws(variables = "theta_star", format = "df")[["theta_star"]]

summary(theta_draws_map)
hist(theta_draws_map, breaks = 30, main = expression("MAP predictive draws for" * theta[star]), xlab = "theta_star")

# Compare the two predictive priors

par(mfrow = c(1, 2))
hist(theta_star_draws, breaks = 30, main = expression("Irregular AR(
1) predictive draws for" * theta[star]), xlab = "theta_star")
hist(theta_draws_map, breaks = 30, main = expression("MAP predictive draws for
" * theta[star]), xlab = "theta_star")

par(mfrow = c(1, 1))
plot(density(theta_star_draws), main = expression("Predictive draws for" * theta[star]), xlab = "theta_star")
lines(density(theta_draws_map), col = "red", lwd = 2)
legend("topright", legend = c("Irregular AR(1)", "MAP"), col
       = c("black", "red"), lwd = 2)       

# using the predictive draws as priors for the current trial analysis

mix_prior_map <- automixfit(theta_draws_map)

alpha_m_map <- mix_prior_map["m",]
alpha_s_map <- mix_prior_map["s",]
alpha_w_map <- mix_prior_map["w",]

# Inspect mixture
print(mix_prior_map)
plot(mix_prior_map)

# We could then re-run the current trial analysis using the MAP predictive prior instead of the irregular AR(1) predictive prior, and compare results. This would involve updating the data list for the current trial model with the new mixture parameters from mix_prior_map, and re-fitting the model.

stan_data_current_map <- list(
  N = nrow(current_df),
  y = current_df$y,
  z = current_df$arm,
  M = length(alpha_w_map),
  w_mix = alpha_w_map,
  mu_mix = alpha_m_map,
  sd_mix = alpha_s_map
)

mod_current_map <- cmdstan_model("current_model.stan")


fit_current_map <- mod_current$sample(
  data = stan_data_current_map,
  seed = 5678,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 200
)


fit_current_map$summary()

# robustify the MAP prior by adding a vague component

mix_prior_rmap <- robustify(mix_prior_map, weight = 0.2, mean = 0, sigma = 1)

alpha_m_rmap <- mix_prior_rmap["m",]
alpha_s_rmap <- mix_prior_rmap["s",]
alpha_w_rmap <- mix_prior_rmap["w",]

print(rubst_mix)
plot(rubst_mix)

stan_data_current_rmap <- list(
  N = nrow(current_df),
  y = current_df$y,
  z = current_df$arm,
  M = length(alpha_w_rmap),
  w_mix = alpha_w_rmap,
  mu_mix = alpha_m_rmap,
  sd_mix = alpha_s_rmap
)

#mod_current_rmap <- cmdstan_model("current_model.stan")


fit_current_rmap <- mod_current$sample(
  data = stan_data_current_rmap,
  seed = 5678,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 200
)

fit_current_rmap$summary()


