td_stage1_stan <- '
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
'
writeLines(td_stage1_stan, "td_hist_stage1.stan")



map_stage1_stan <- '
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
'
writeLines(map_stage1_stan, "map_stage1.stan")


current_mix_stan <- '
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
'
writeLines(current_mix_stan, "current_mix_model.stan")


current_nb_stan <- '
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
'
writeLines(current_nb_stan, "current_nb_model.stan")

library(cmdstanr)
library(RBesT)
library(posterior)
library(dplyr)

mod_td_stage1  <- cmdstan_model("td_hist_stage1.stan")
mod_map_stage1 <- cmdstan_model("map_stage1.stan")
mod_current_mix <- cmdstan_model("current_mix_model.stan")
mod_current_nb  <- cmdstan_model("current_nb_model.stan")


##### UTILITY FUNCIONS #####################

generate_hist_means <- function(mu_star, time, psi = 0, drift = c("none", "constant", "linear", "step", "ar1"),
                                phi = 0.8, omega = 0.5) {
  drift <- match.arg(drift)
  H <- length(time)
  
  d <- numeric(H)
  
  if (drift == "none") {
    d <- rep(0, H)
  }
  
  if (drift == "constant") {
    d <- rep(psi, H)
  }
  
  if (drift == "linear") {
    d <- psi * (max(time) - time) / (max(time) - min(time))
  }
  
  if (drift == "step") {
    d <- ifelse(seq_len(H) <= ceiling(H / 2), psi, 0)
  }
  
  if (drift == "ar1") {
    d[1] <- rnorm(1, mean = psi, sd = omega)
    for (h in 2:H) {
      delta_h <- time[h] - time[h - 1]
      d[h] <- phi^delta_h * d[h - 1] + rnorm(1, mean = 0, sd = omega)
    }
  }
  
  mu_hist_true <- mu_star + d
  return(mu_hist_true)
}


generate_hist_summary <- function(mu_hist_true, time, n_hist = 100, sigma_hist = 2) {
  H <- length(mu_hist_true)
  if (length(n_hist) == 1) n_hist <- rep(n_hist, H)
  
  ybar_hist <- rnorm(H, mean = mu_hist_true, sd = sigma_hist / sqrt(n_hist))
  sd_hist <- rep(sigma_hist, H)
  se_hist <- sd_hist / sqrt(n_hist)
  
  data.frame(
    study_id = paste0("H", seq_len(H)),
    time = time,
    n_hist = n_hist,
    mu_hist_true = mu_hist_true,
    ybar_hist = ybar_hist,
    sd_hist = sd_hist,
    se_hist = se_hist
  )
}


generate_current_trial <- function(n_cur = 200, r = 0.7, mu_star = 10.5, theta_trt = 0.5, sigma = 2) {
  n_trt <- round(n_cur * r)
  n_ctrl <- n_cur - n_trt
  z <- c(rep(0, n_ctrl), rep(1, n_trt))
  y <- rnorm(n_cur, mean = mu_star + theta_trt * z, sd = sigma)
  
  data.frame(
    id = seq_len(n_cur),
    arm = z,
    y = y
  )
}


fit_td_prior <- function(hist_df, t_star, mod, seed = 1234, chains = 4, iter_warmup = 1000, iter_sampling = 1000) {
  stan_data <- list(
    H = nrow(hist_df),
    ybar = hist_df$ybar_hist,
    se = hist_df$se_hist,
    time = hist_df$time,
    t_star = t_star
  )
  
  fit <- mod$sample(
    data = stan_data,
    seed = seed,
    chains = chains,
    parallel_chains = chains,
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    refresh = 0
  )
  
  theta_star_draws <- fit$draws(variables = "theta_star", format = "df")[["theta_star"]]
  mix_prior <- automixfit(theta_star_draws)
  
  list(fit = fit, theta_star_draws = theta_star_draws, mix_prior = mix_prior)
}


fit_map_prior <- function(hist_df, mod, seed = 1234, chains = 4, iter_warmup = 1000, iter_sampling = 1000) {
  stan_data <- list(
    H = nrow(hist_df),
    ybar = hist_df$ybar_hist,
    se = hist_df$se_hist
  )
  
  fit <- mod$sample(
    data = stan_data,
    seed = seed,
    chains = chains,
    parallel_chains = chains,
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    refresh = 0
  )
  
  theta_star_draws <- fit$draws(variables = "theta_star", format = "df")[["theta_star"]]
  mix_prior <- automixfit(theta_star_draws)
  
  list(fit = fit, theta_star_draws = theta_star_draws, mix_prior = mix_prior)
}

robustify_prior <- function(mix_prior, weight = 0.2, mean = 0, sigma = 10) {
  robustify(mix_prior, weight = weight, mean = mean, sigma = sigma)
}

extract_mix_components <- function(mix_prior) {
  list(
    w_mix = as.numeric(mix_prior["w", ]),
    mu_mix = as.numeric(mix_prior["m", ]),
    sd_mix = as.numeric(mix_prior["s", ])
  )
}


fit_current_mixture <- function(current_df, mix_prior, mod, seed = 5678, chains = 4,
                                iter_warmup = 1000, iter_sampling = 1000) {
  mix_comp <- extract_mix_components(mix_prior)
  
  stan_data <- list(
    N = nrow(current_df),
    y = current_df$y,
    z = current_df$arm,
    M = length(mix_comp$w_mix),
    w_mix = mix_comp$w_mix / sum(mix_comp$w_mix),
    mu_mix = mix_comp$mu_mix,
    sd_mix = mix_comp$sd_mix
  )
  
  mod$sample(
    data = stan_data,
    seed = seed,
    chains = chains,
    parallel_chains = chains,
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    refresh = 0
  )
}


fit_current_nb <- function(current_df, mod, seed = 5678, chains = 4,
                           iter_warmup = 1000, iter_sampling = 1000) {
  stan_data <- list(
    N = nrow(current_df),
    y = current_df$y,
    z = current_df$arm
  )
  
  mod$sample(
    data = stan_data,
    seed = seed,
    chains = chains,
    parallel_chains = chains,
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    refresh = 0
  )
}


extract_trial_summary <- function(fit, beta_true, prob_threshold = 0.975) {
  draws <- as_draws_df(fit$draws(c("beta", "theta_star", "sigma")))
  
  beta_draws <- draws$beta
  theta_star_draws <- draws$theta_star
  
  beta_mean <- mean(beta_draws)
  beta_sd   <- sd(beta_draws)
  beta_lwr  <- quantile(beta_draws, 0.025)
  beta_upr  <- quantile(beta_draws, 0.975)
  
  theta_star_mean <- mean(theta_star_draws)
  theta_star_sd   <- sd(theta_star_draws)
  
  p_benefit <- mean(beta_draws < 0)
  decision  <- as.integer(p_benefit > prob_threshold)
  
  tibble::tibble(
    beta_true = beta_true,
    beta_mean = beta_mean,
    beta_sd = beta_sd,
    beta_lwr = beta_lwr,
    beta_upr = beta_upr,
    theta_star_mean = theta_star_mean,
    theta_star_sd = theta_star_sd,
    p_benefit = p_benefit,
    decision = decision
  )
}


run_one_simulation <- function(
    scenario_id = "S1",
    H = 5,
    time = c(0, 2, 4, 7, 10),
    delta_star = 1,
    n_hist = 100,
    n_cur = 200,
    r = 0.7,
    mu_star = 10.5,
    theta_trt = 0,
    sigma = 2,
    psi = 0,
    drift = c("none", "constant", "linear", "step", "ar1"),
    phi = 0.8,
    omega = 0.5,
    robust_weight = 0.2,
    mod_td_stage1,
    mod_map_stage1,
    mod_current_mix,
    mod_current_nb
) {
  drift <- match.arg(drift)
  
  # Step 1: Generate historical truth
  mu_hist_true <- generate_hist_means(
    mu_star = mu_star,
    time = time,
    psi = psi,
    drift = drift,
    phi = phi,
    omega = omega
  )
  
  # Step 2: Historical summaries
  hist_df <- generate_hist_summary(
    mu_hist_true = mu_hist_true,
    time = time,
    n_hist = n_hist,
    sigma_hist = sigma
  )
  
  t_star <- max(time) + delta_star
  
  # Step 3: First-stage fits
  fit_td <- fit_td_prior(hist_df = hist_df, t_star = t_star, mod = mod_td_stage1)
  fit_map <- fit_map_prior(hist_df = hist_df, mod = mod_map_stage1)
  
  # Robustified priors
  mix_td <- fit_td$mix_prior
  mix_map <- fit_map$mix_prior
  mix_rtd <- robustify_prior(mix_td, weight = robust_weight, mean = 0, sigma = 10)
  mix_rmap <- robustify_prior(mix_map, weight = robust_weight, mean = 0, sigma = 10)
  
  # Step 4: Current trial
  current_df <- generate_current_trial(
    n_cur = n_cur,
    r = r,
    mu_star = mu_star,
    theta_trt = theta_trt,
    sigma = sigma
  )
  
  # Step 5: Fit current trial models
  fit_nb   <- fit_current_nb(current_df, mod_current_nb)
  fit_map2 <- fit_current_mixture(current_df, mix_map, mod_current_mix)
  fit_rmap <- fit_current_mixture(current_df, mix_rmap, mod_current_mix)
  fit_td2  <- fit_current_mixture(current_df, mix_td, mod_current_mix)
  fit_rtd  <- fit_current_mixture(current_df, mix_rtd, mod_current_mix)
  
  # Step 6: Extract summaries
  out_nb <- extract_trial_summary(fit_nb, beta_true = theta_trt) |>
    dplyr::mutate(model = "No borrowing", scenario = scenario_id)
  
  out_map <- extract_trial_summary(fit_map2, beta_true = theta_trt) |>
    dplyr::mutate(model = "MAP prior", scenario = scenario_id)
  
  out_rmap <- extract_trial_summary(fit_rmap, beta_true = theta_trt) |>
    dplyr::mutate(model = "Robust MAP prior", scenario = scenario_id)
  
  out_td <- extract_trial_summary(fit_td2, beta_true = theta_trt) |>
    dplyr::mutate(model = "Time-dependent prior", scenario = scenario_id)
  
  out_rtd <- extract_trial_summary(fit_rtd, beta_true = theta_trt) |>
    dplyr::mutate(model = "Robustified Time-dependent prior", scenario = scenario_id)
  
  dplyr::bind_rows(out_nb, out_map, out_rmap, out_td, out_rtd)
}


compute_oc <- function(results_df) {
  results_df |>
    dplyr::group_by(scenario, model) |>
    dplyr::summarise(
      bias = mean(beta_mean - beta_true),
      rmse = sqrt(mean((beta_mean - beta_true)^2)),
      coverage = mean(beta_lwr <= beta_true & beta_upr >= beta_true),
      type1_error = mean(decision[beta_true == 0], na.rm = TRUE),
      power = mean(decision[beta_true != 0], na.rm = TRUE),
      mean_post_sd = mean(beta_sd),
      .groups = "drop"
    )
}


compute_oc <- function(results_df) {
  results_df |>
    dplyr::group_by(scenario, model) |>
    dplyr::summarise(
      bias = mean(beta_mean - beta_true),
      rmse = sqrt(mean((beta_mean - beta_true)^2)),
      coverage = mean(beta_lwr <= beta_true & beta_upr >= beta_true),
      mean_post_sd = mean(beta_sd),
      rejection_rate = mean(decision),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      type1_error = ifelse(grepl("null", scenario, ignore.case = TRUE), rejection_rate, NA_real_),
      power = ifelse(!grepl("null", scenario, ignore.case = TRUE), rejection_rate, NA_real_)
    )
}


set.seed(2026)

one_run <- run_one_simulation(
  scenario_id = "S_linear",
  time = c(0, 2, 4, 7, 10),
  delta_star = 1,
  n_hist = 100,
  n_cur = 200,
  r = 0.7,
  mu_star = 10.5,
  theta_trt = 0.5,
  sigma = 2,
  psi = 1,
  drift = "linear",
  mod_td_stage1 = mod_td_stage1,
  mod_map_stage1 = mod_map_stage1,
  mod_current_mix = mod_current_mix,
  mod_current_nb = mod_current_nb
)

one_run

run_simulation_study <- function(R = 100, ...) {
  out_list <- vector("list", R)
  
  for (r in seq_len(R)) {
    out_list[[r]] <- run_one_simulation(...)
    message("Completed replicate ", r, " / ", R)
  }
  
  dplyr::bind_rows(out_list, .id = "replicate")
}
