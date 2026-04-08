# =============================================================================
# workflow.R
#
# FULL THREE-STEP BAYESIAN HISTORICAL BORROWING WORKFLOW
#
# Step 1 [Stan]  : Sample from the non-stationary AR(1) posterior predictive
#                  prior for theta_0 given K historical sufficient statistics.
# Step 2 [RBesT] : Fit a conjugate mixture-of-normals approximation to the
#                  theta_0 samples using RBesT::automixfit().
# Step 3 [Stan]  : Compute the posterior of theta_0 for the new trial using
#                  the mixture prior from Step 2 and the current trial data.
#
# DEPENDENCIES
#   install.packages(c("cmdstanr", "RBesT", "posterior", "bayesplot", "ggplot2"))
#   cmdstanr::install_cmdstan()   # one-time installation
# =============================================================================

library(cmdstanr)
library(RBesT)
library(posterior)
library(bayesplot)
library(ggplot2)

# ── Colour palette and ggplot theme ──────────────────────────────────────────
theme_set(theme_bw(base_size = 13))
col_hist    <- "#2166AC"   # blue  – historical
col_pred    <- "#D6604D"   # red   – predictive prior
col_post    <- "#4DAC26"   # green – new trial posterior
col_mixture <- "#762A83"   # purple – mixture approximation

# =============================================================================
# 0.  DATA SPECIFICATION
#     Replace this section with your actual data.
# =============================================================================

# ── Historical trial sufficient statistics ────────────────────────────────────
# Each row: one historical trial.
# s2_k = sigma^2 / n_k  (sampling variance of the sample mean)
# Here sigma^2 is the known within-trial variance.

K       <- 5L                                # number of historical trials
sigma2  <- 4.0                               # known within-trial variance

hist_data <- data.frame(
  trial   = 1:K,
  t_hist  = c(2015, 2016, 2018, 2020, 2022),  # calendar year
  n_k     = c(  50,   60,   80,   70,   90),  # sample sizes
  ybar_k  = c( 1.8,  2.1,  2.0,  2.3,  2.2)  # observed sample means
)
hist_data$s2_k <- sigma2 / hist_data$n_k

# ── Current trial (Step 3 only) ───────────────────────────────────────────────
t0    <- 2024            # current trial time
n0    <- 100L            # current trial sample size
ybar0 <- 2.15            # observed sample mean (current trial)
s0sq  <- sigma2 / n0     # sampling variance

# =============================================================================
# 1.  STEP 1: NON-STATIONARY AR(1) PREDICTIVE PRIOR
#     Fit the hierarchical Stan model to obtain posterior samples of theta_0.
# =============================================================================

# ── 1a. Compile model ─────────────────────────────────────────────────────────
cat("\n[Step 1] Compiling AR(1) non-stationary predictive prior model...\n")
mod_pred <- cmdstan_model(
  "ar1map.stan",
  cpp_options = list(stan_threads = TRUE)   # optional: within-chain parallelism
)

# ── 1b. Assemble Stan data list ───────────────────────────────────────────────
# Hyperprior specification:
#   mu1    ~ N(mu1_m, mu1_s^2)     : prior on initial mean
#   sig1sq ~ InvGamma(a1, b1)      : prior on initial variance
#   mu     ~ N(mu_m,  mu_s^2)      : prior on long-run mean
#   tau2   ~ InvGamma(atau, btau)  : prior on innovation variance
#   rho    ~ Beta(rho_a, rho_b)    : prior on autocorrelation

# Calibration guidance:
#   - mu1_m / mu_m : set to a plausible historical mean (e.g. prior meta-analytic estimate)
#   - mu1_s / mu_s : wide relative to expected range of true effects
#   - sig1sq_a/b, tau2_a/b : InvGamma(2, sigma_guess^2) concentrates near sigma_guess^2
#   - rho_a, rho_b : (2, 1) favours positive autocorrelation; (1, 1) is uniform on (0,1)

stan_data_pred <- list(
  K         = K,
  ybar      = hist_data$ybar_k,
  s2        = hist_data$s2_k,
  t_hist    = hist_data$t_hist,
  t0        = t0,

  # Hyperpriors
  mu1_m     =  2.0,    # prior mean for mu1
  mu1_s     =  2.0,    # prior SD   for mu1
  sig1sq_a  =  2.0,    # InvGamma shape for sig1sq
  sig1sq_b  =  1.0,    # InvGamma scale for sig1sq  => E[sig1sq] = b/(a-1) = 1.0
  mu_m      =  2.0,    # prior mean for long-run mean
  mu_s      =  2.0,    # prior SD   for long-run mean
  tau2_a    =  2.0,    # InvGamma shape for tau2
  tau2_b    =  0.5,    # InvGamma scale for tau2    => E[tau2] = 0.5
  rho_a     =  2.0,    # Beta shape: favours rho close to 1
  rho_b     =  1.0,

  jitter    =  1e-6    # diagonal jitter for PD stability
)

# ── 1c. Run MCMC ──────────────────────────────────────────────────────────────
cat("[Step 1] Sampling...\n")
fit_pred <- mod_pred$sample(
  data            = stan_data_pred,
  seed            = 42L,
  chains          = 4L,
  parallel_chains = 4L,
  iter_warmup     = 1000L,
  iter_sampling   = 2000L,
  refresh         = 500L,
  adapt_delta     = 0.95,
  max_treedepth   = 12L
)

# ── 1d. Diagnostics ───────────────────────────────────────────────────────────
cat("\n[Step 1] Diagnostics:\n")
fit_pred$cmdstan_diagnose()

# Trace plots for key hyperparameters
draws_pred <- fit_pred$draws(format = "draws_df")
params_to_check <- c("mu1", "sig1sq", "mu", "rho", "tau2", "theta0")
print(
  mcmc_trace(fit_pred$draws(params_to_check),
             facet_args = list(ncol = 2)) +
    ggtitle("Step 1: Trace plots – hyperparameters and theta_0")
)

# Summary of hyperparameters
summ_pred <- fit_pred$summary(params_to_check)
cat("\n--- Posterior summary (hyperparameters + theta_0) ---\n")
print(summ_pred, digits = 3)

# ── 1e. Extract theta_0 samples ───────────────────────────────────────────────
theta0_samples <- as.vector(fit_pred$draws("theta0", format = "matrix"))
cat(sprintf("\n[Step 1] Extracted %d theta_0 samples.\n", length(theta0_samples)))

# Density of predictive prior
p_pred_prior <- ggplot(data.frame(theta0 = theta0_samples), aes(x = theta0)) +
  geom_density(fill = col_pred, alpha = 0.4, colour = col_pred, linewidth = 0.8) +
  labs(title = "Step 1: Posterior predictive prior for theta_0",
       x = expression(theta[0]), y = "Density") +
  theme_bw()
print(p_pred_prior)

# =============================================================================
# 2.  STEP 2: MIXTURE APPROXIMATION VIA RBesT::automixfit()
#     Fit a conjugate mixture of normals to the theta_0 posterior samples.
#     This mixture becomes the informative prior for the new trial.
# =============================================================================
cat("\n[Step 2] Fitting mixture prior via RBesT::automixfit()...\n")

# automixfit() fits a normal mixture via EM, selecting the number of
# components automatically by minimising a model-selection criterion.
# Key arguments:
#   Nc   : NULL (auto-select) or integer vector of component counts to try
#   type : "norm" for normal mixture (continuous endpoint)
#   eps  : convergence tolerance

mix_prior <- automixfit(
  sample = theta0_samples,
  type   = "norm",
  Nc     = 1:5,        # try 1 to 5 components; automixfit picks the best
  eps    = 1e-6
)

cat("\n--- Mixture prior ---\n")
print(mix_prior)
print(summary(mix_prior))

# Extract mixture parameters for Stan
# A mixnorm object is a 3 x C matrix: row 1 = weights, row 2 = means, row 3 = SDs
C_mix      <- ncol(mix_prior)
w_mix      <- mix_prior[1, ]          # weights
means_mix  <- mix_prior[2, ]          # means
sds_mix    <- mix_prior[3, ]          # standard deviations

cat(sprintf("\n[Step 2] Mixture has %d component(s):\n", C_mix))
mix_df <- data.frame(
  Component = seq_len(C_mix),
  Weight    = round(w_mix,    4),
  Mean      = round(means_mix, 4),
  SD        = round(sds_mix,   4)
)
print(mix_df)

# ── 2b. Mixture fit diagnostics ───────────────────────────────────────────────
# Overlay: raw predictive prior density vs mixture approximation

theta_grid  <- seq(min(theta0_samples) - 1, max(theta0_samples) + 1, length.out = 500)
mix_density <- dmixnorm(theta_grid, mix_prior)

p_mix_fit <- ggplot() +
  geom_density(data = data.frame(theta0 = theta0_samples),
               aes(x = theta0),
               fill = col_pred, alpha = 0.3, colour = col_pred, linewidth = 0.8) +
  geom_line(data = data.frame(x = theta_grid, y = mix_density),
            aes(x = x, y = y),
            colour = col_mixture, linewidth = 1.0, linetype = "dashed") +
  labs(title  = "Step 2: Predictive prior (shaded) vs mixture approximation (dashed)",
       x      = expression(theta[0]),
       y      = "Density") +
  annotate("text", x = Inf, y = Inf,
           label  = sprintf("%d-component normal mixture", C_mix),
           hjust  = 1.1, vjust = 1.5,
           colour = col_mixture, size = 4)
print(p_mix_fit)

# ── 2c. Effective sample size of the mixture prior ────────────────────────────
ess_mix <- ess(mix_prior)
cat(sprintf("\n[Step 2] Effective sample size of mixture prior: %.1f\n", ess_mix))

# =============================================================================
# 3.  STEP 3: NEW TRIAL POSTERIOR
#     Use the mixture prior and current trial sufficient statistics to obtain
#     the posterior of theta_0 via Stan.
# =============================================================================
cat("\n[Step 3] Compiling new trial posterior model...\n")
mod_post <- cmdstan_model("new_trial_posterior.stan")

# ── 3a. Stan data ─────────────────────────────────────────────────────────────
stan_data_post <- list(
  ybar0         = ybar0,
  s0sq          = s0sq,

  # Mixture prior from Step 2
  C             = C_mix,
  w             = as.numeric(w_mix),
  mix_means     = as.numeric(means_mix),
  mix_sds       = as.numeric(sds_mix),

  # Robust mixture option (Schmidli et al. 2014):
  #   robust_weight = 0   => use informative prior as-is
  #   robust_weight = 0.1 => 10% weight on vague N(0, 100^2) component
  robust_weight = 0.1,
  vague_sd      = 100.0
)

# ── 3b. Sample ────────────────────────────────────────────────────────────────
cat("[Step 3] Sampling...\n")
fit_post <- mod_post$sample(
  data            = stan_data_post,
  seed            = 42L,
  chains          = 4L,
  parallel_chains = 4L,
  iter_warmup     = 1000L,
  iter_sampling   = 2000L,
  refresh         = 500L,
  adapt_delta     = 0.95
)

# ── 3c. Diagnostics ───────────────────────────────────────────────────────────
cat("\n[Step 3] Diagnostics:\n")
fit_post$cmdstan_diagnose()

# ── 3d. Summarise posterior ───────────────────────────────────────────────────
summ_post <- fit_post$summary("theta0")
cat("\n--- Posterior summary for theta_0 (new trial) ---\n")
print(summ_post, digits = 4)

theta0_post <- as.vector(fit_post$draws("theta0", format = "matrix"))

# =============================================================================
# 4.  COMBINED SUMMARY PLOT
#     Overlay: predictive prior | mixture approx | new trial posterior
# =============================================================================

post_density  <- density(theta0_post)
prior_density <- density(theta0_samples)
mix_y         <- dmixnorm(theta_grid, mix_prior)

p_combined <- ggplot() +
  # Predictive prior (Step 1)
  geom_density(data = data.frame(theta0 = theta0_samples),
               aes(x = theta0, fill = "Predictive prior", colour = "Predictive prior"),
               alpha = 0.25, linewidth = 0.8) +
  # Mixture approximation (Step 2)
  geom_line(data = data.frame(x = theta_grid, y = mix_y),
            aes(x = x, y = y, colour = "Mixture approx", fill = "Mixture approx"),
            linewidth = 1.0, linetype = "dashed") +
  # New trial posterior (Step 3)
  geom_density(data = data.frame(theta0 = theta0_post),
               aes(x = theta0, fill = "New trial posterior", colour = "New trial posterior"),
               alpha = 0.30, linewidth = 0.9) +
  # Current trial data marker
  geom_vline(xintercept = ybar0, colour = "black", linetype = "dotted", linewidth = 0.9) +
  annotate("text", x = ybar0 + 0.02, y = Inf, vjust = 1.4, hjust = 0,
           label = sprintf("bar(y)[0] == %.2f", ybar0), parse = TRUE, size = 3.5) +
  scale_fill_manual(
    name   = NULL,
    values = c("Predictive prior"    = col_pred,
               "Mixture approx"      = col_mixture,
               "New trial posterior" = col_post)
  ) +
  scale_colour_manual(
    name   = NULL,
    values = c("Predictive prior"    = col_pred,
               "Mixture approx"      = col_mixture,
               "New trial posterior" = col_post)
  ) +
  labs(
    title    = "Bayesian Historical Borrowing: Non-Stationary AR(1) Prior",
    subtitle = sprintf(
      "%d historical trials | %d-component mixture prior | n[0] = %d, bar(y)[0] = %.2f",
      K, C_mix, n0, ybar0),
    x = expression(theta[0]),
    y = "Density"
  ) +
  theme(legend.position = "top")

print(p_combined)
ggsave("borrowing_summary.pdf", p_combined, width = 9, height = 5.5)

# =============================================================================
# 5.  NUMERICAL SUMMARY TABLE
# =============================================================================

# Posterior credible intervals
q_prior <- quantile(theta0_samples, c(0.025, 0.5, 0.975))
q_post  <- quantile(theta0_post,    c(0.025, 0.5, 0.975))

summary_table <- data.frame(
  Quantity = c(
    "Predictive prior (Step 1)",
    "New trial posterior (Step 3)"
  ),
  Mean   = c(mean(theta0_samples), mean(theta0_post)),
  SD     = c(sd(theta0_samples),   sd(theta0_post)),
  Q2.5   = c(q_prior[1],           q_post[1]),
  Median = c(q_prior[2],           q_post[2]),
  Q97.5  = c(q_prior[3],           q_post[3])
)

cat("\n=== FINAL SUMMARY TABLE ===\n")
print(summary_table, digits = 4, row.names = FALSE)

cat(sprintf(
  "\nMixture prior ESS: %.1f  |  Current trial n0: %d\n",
  ess_mix, n0
))

# =============================================================================
# HELPER FUNCTION: RUN THE FULL PIPELINE IN ONE CALL
# =============================================================================

#' run_historical_borrowing
#'
#' Convenience wrapper executing the complete three-step pipeline.
#'
#' @param hist_df    data.frame with columns: t_hist, n_k, ybar_k
#' @param sigma2     known within-trial variance
#' @param t0         current trial time
#' @param n0         current trial sample size
#' @param ybar0      current trial sample mean
#' @param hyperpriors named list of hyperprior parameters (see stan_data_pred above)
#' @param robust_wt  robust mixture weight (default 0.1)
#' @param seed       random seed
#' @param chains     number of MCMC chains
#' @param iter_samp  number of sampling iterations per chain
#' @return list with: fit_pred, mix_prior, fit_post, summary_table
run_historical_borrowing <- function(
    hist_df,
    sigma2,
    t0,
    n0,
    ybar0,
    hyperpriors  = list(
      mu1_m    =  0,    mu1_s    = 10,
      sig1sq_a =  2,    sig1sq_b =  1,
      mu_m     =  0,    mu_s     = 10,
      tau2_a   =  2,    tau2_b   =  1,
      rho_a    =  2,    rho_b    =  1
    ),
    robust_wt    = 0.1,
    vague_sd     = 100,
    seed         = 42L,
    chains       = 4L,
    iter_samp    = 2000L
) {
  K    <- nrow(hist_df)
  s2_k <- sigma2 / hist_df$n_k
  s0sq <- sigma2 / n0

  # ── Step 1: Predictive prior ───────────────────────────────────────────────
  mod1 <- cmdstan_model("ar1ns_predictive_prior.stan")
  fit1 <- mod1$sample(
    data = c(
      list(K = K, ybar = hist_df$ybar_k, s2 = s2_k,
           t_hist = hist_df$t_hist, t0 = t0, jitter = 1e-6),
      hyperpriors
    ),
    seed = seed, chains = chains,
    parallel_chains = chains,
    iter_warmup  = 1000L,
    iter_sampling = iter_samp,
    refresh = 0, adapt_delta = 0.95, max_treedepth = 12L
  )
  theta0_samp <- as.vector(fit1$draws("theta0", format = "matrix"))

  # ── Step 2: Mixture approximation ─────────────────────────────────────────
  mix <- automixfit(theta0_samp, type = "norm", Nc = 1:5, eps = 1e-6)
  C_m <- ncol(mix)

  # ── Step 3: New trial posterior ────────────────────────────────────────────
  mod2 <- cmdstan_model("new_trial_posterior.stan")
  fit2 <- mod2$sample(
    data = list(
      ybar0         = ybar0,
      s0sq          = s0sq,
      C             = C_m,
      w             = as.numeric(mix[1, ]),
      mix_means     = as.numeric(mix[2, ]),
      mix_sds       = as.numeric(mix[3, ]),
      robust_weight = robust_wt,
      vague_sd      = vague_sd
    ),
    seed = seed, chains = chains,
    parallel_chains = chains,
    iter_warmup   = 1000L,
    iter_sampling = iter_samp,
    refresh = 0, adapt_delta = 0.95
  )

  theta0_post <- as.vector(fit2$draws("theta0", format = "matrix"))

  tbl <- data.frame(
    Quantity = c("Predictive prior", "New trial posterior"),
    Mean     = c(mean(theta0_samp), mean(theta0_post)),
    SD       = c(sd(theta0_samp),   sd(theta0_post)),
    Q2.5     = c(quantile(theta0_samp, 0.025), quantile(theta0_post, 0.025)),
    Q97.5    = c(quantile(theta0_samp, 0.975), quantile(theta0_post, 0.975))
  )

  list(
    fit_pred      = fit1,
    theta0_pred   = theta0_samp,
    mix_prior     = mix,
    fit_post      = fit2,
    theta0_post   = theta0_post,
    summary_table = tbl
  )
}

# Example call:
# result <- run_historical_borrowing(hist_df = hist_data, sigma2 = sigma2,
#                                    t0 = t0, n0 = n0, ybar0 = ybar0)
# print(result$summary_table)
# print(result$mix_prior)
