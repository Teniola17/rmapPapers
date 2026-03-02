# ------------------------------------------------------------
# Simulation of borrowing methods for binary outcomes
# Based on Viele et al.
# ------------------------------------------------------------

set.seed(2025)
library(stats)
library(MASS)       # for mvrnorm if needed later
library(tidyverse)

#---------------------------
# Simulation parameters
#---------------------------
n_hist <- 100               # historical sample size
y_hist <- 65                # historical responses
p_hist <- y_hist / n_hist   # historical rate = 0.65

n_curr  <- 200              # per-arm (treatment/control)
alpha   <- 0.025            # nominal type I error for trial
n_sim   <- 1000             # Monte Carlo replicates

p0_seq  <- seq(0.50, 0.8, by = 0.002)  # current control true rates
delta   <- 0.12                         # treatment improvement
methods <- c("separate", "pool", "single", "test_then_pool", "powerprior", "hierarchical")

#---------------------------
# Helper functions
#---------------------------

# 1. Separate analysis (Fisher exact test)
analyze_separate <- function(y0, yT) {
  mat <- matrix(c(yT, n_curr - yT, y0, n_curr - y0), ncol = 2, byrow = TRUE)
  pval <- fisher.test(mat, alternative = "greater")$p.value
  decision <- (pval < alpha)
  list(decision = decision, est_c = y0 / n_curr)
}

# 2. Pooling method
analyze_pool <- function(y0, yT) {
  y0_pool <- y0 + y_hist
  n0_pool <- n_curr + n_hist
  mat <- matrix(c(yT, n_curr - yT, y0_pool, n0_pool - y0_pool), ncol = 2, byrow = TRUE)
  pval <- fisher.test(mat, alternative = "greater")$p.value
  decision <- (pval < alpha)
  list(decision = decision, est_c = y0_pool / n0_pool)
}

# 3. Single-arm trial (compare to fixed benchmark)
analyze_single <- function(yT) {
  pval <- binom.test(yT, n_curr, p_hist, alternative = "greater")$p.value
  decision <- (pval < alpha)
  list(decision = decision, est_c = p_hist)
}

# 4. Test-then-pool (dynamic "all-or-none")
analyze_test_then_pool <- function(y0, yT, alpha_test = 0.10) {
  pval_eq <- prop.test(c(y0, y_hist), c(n_curr, n_hist))$p.value
  if (pval_eq > alpha_test) {
    res <- analyze_pool(y0, yT)
    res$borrowed <- n_hist
  } else {
    res <- analyze_separate(y0, yT)
    res$borrowed <- 0
  }
  res
}

# 5. Power prior (fixed downweight)
analyze_power_prior <- function(y0, yT, a = 0.4) {
  # prior from historical: Beta(a*y_hist + 0.001, a*(n_hist - y_hist) + 0.001)
  alpha_prior <- 0.001 + a * y_hist
  beta_prior  <- 0.001 + a * (n_hist - y_hist)
  # posterior for control and treatment
  post_c <- rbeta(5000, alpha_prior + y0, beta_prior + n_curr - y0)
  post_t <- rbeta(5000, 0.001 + yT, 0.001 + n_curr - yT)
  prob <- mean(post_t > post_c)
  decision <- (prob > 0.975)
  list(decision = decision, est_c = mean(post_c))
}

# 6. Simple hierarchical borrowing (approximate)
# Uses logit-normal prior on control
analyze_hierarchical <- function(y0, yT, beta_prior = 0.01) {
  # Priors
  logit <- function(p) log(p / (1 - p))
  invlogit <- function(x) exp(x) / (1 + exp(x))
  
  mu <- logit(p_hist)
  tau <- sqrt(beta_prior) # smaller beta_prior = stronger borrowing
  
  # Sample current control parameter from prior predictive distribution
  gamma_c <- rnorm(5000, mean = mu, sd = tau)
  p0_prior <- invlogit(gamma_c)
  
  # Posterior predictive update (rough dynamic borrowing)
  p0_post <- rbeta(5000, y0 + 1, n_curr - y0 + 1)
  pT_post <- rbeta(5000, yT + 1, n_curr - yT + 1)
  
  prob <- mean(pT_post > p0_post)
  decision <- (prob > 0.975)
  list(decision = decision, est_c = mean(p0_post))
}

#---------------------------
# Simulation loop
#---------------------------

simulate_trial <- function(p0, delta, method) {
  y0 <- rbinom(1, n_curr, p0)
  yT <- rbinom(1, n_curr, p0 + delta)
  switch(method,
         separate = analyze_separate(y0, yT),
         pool = analyze_pool(y0, yT),
         single = analyze_single(yT),
         test_then_pool = analyze_test_then_pool(y0, yT, alpha_test = 0.10),
         powerprior = analyze_power_prior(y0, yT, a = 0.4),
         hierarchical = analyze_hierarchical(y0, yT, beta_prior = 0.01))
}

#---------------------------
# This is the simulation running section
#---------------------------

results <- expand.grid(p0 = p0_seq, method = methods)
results$type1 <- results$power <- results$mse <- NA
t1 <- Sys.time()
for (i in seq_len(nrow(results))) {
  p0 <- results$p0[i]; m <- results$method[i]
  type1_vec <- power_vec <- numeric(n_sim)
  est_vec   <- numeric(n_sim)
  
  for (k in seq_len(n_sim)) {
    res_null <- simulate_trial(p0, delta = 0, method = m)
    res_alt  <- simulate_trial(p0, delta = 0.12, method = m)
    type1_vec[k] <- res_null$decision
    power_vec[k] <- res_alt$decision
    est_vec[k]   <- res_null$est_c
  }
  
  results$type1[i] <- mean(type1_vec)
  results$power[i] <- mean(power_vec)
  results$mse[i]   <- mean((est_vec - p0)^2)
  cat(sprintf("Done: p0=%.2f, method=%s\n", p0, m))
}