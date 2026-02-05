library(MASS)

# --- 1. SETUP PARAMETERS ---
n_hist <- 200      # Patients per historical study
n_trial <- 500     # Patients in current trial
H <- 5             # Number of historical studies
p_cont <- 50       # Number of continuous features
p_bin <- 5         # Number of binary biomarkers
rho <- 0.8         # Correlation for collinearity
global_hist_drift <- 0  # Pooled shift applied to historical studies vs current (set >0 or <0 to simulate drift)

# --- 2. FUNCTION TO GENERATE FEATURES (X) ---
# Creates shared covariate structure across all studies
generate_x <- function(n) {
  # Collinear continuous features
  sigma <- toeplitz(rho^(0:(p_cont-1)))
  X_cont <- mvrnorm(n, mu = rep(0, p_cont), Sigma = sigma)
  
  # Binary Biomarkers (e.g., DNA markers)
  X_bin <- matrix(rbinom(n * p_bin, 1, 0.3), nrow = n)
  
  colnames(X_cont) <- paste0("Lab_", 1:p_cont)
  colnames(X_bin) <- paste0("Bio_", 1:p_bin)
  return(cbind(X_cont, X_bin))
}

# --- 3. FUNCTION TO GENERATE OUTCOME (Y) ---


generate_y <- function(X, Trt,
                       study_drift = 0,
                       global_hist_drift = 0) {
  
  # 3a. Prognostic Signal
  prog_signal <- (2 * X[, "Lab_1"]) + 
    (3 * X[, "Lab_2"]^2) + 
    (5 * sin(X[, "Lab_3"])) +
    (4 * X[, "Lab_4"] * X[, "Lab_5"])
  
  # 3b. Predictive (Treatment) Signal
  trt_effect <- (5 * Trt) + (12 * Trt * X[, "Bio_1"])
  
  # 3c. Noise
  noise <- rnorm(nrow(X), mean = 0, sd = 2)
  
  y <- prog_signal +
    trt_effect +
    study_drift +          # between-trial (historical only)
    global_hist_drift +    # pooled historical vs current
    noise
  
  return(y)
}


# --- 4. GENERATE HISTORICAL DATA (Placebo Only) ---
historical_data <- lapply(1:H, function(s) {
  X <- generate_x(n_hist)
  Trt <- rep(0, n_hist) # Placebo only
  # Apply both a small per-study drift and the pooled historical drift
  Y <- generate_y(X, Trt,
                  study_drift = runif(1, -2, 2),
                  global_hist_drift = global_hist_drift)
  
  data.frame(Study = paste0(s), Treatment = Trt, Target = Y, X)
})

# --- 5. GENERATE CURRENT TRIAL DATA (Placebo + Treatment) ---
X_trial <- generate_x(n_trial)
Trt_trial <- rbinom(n_trial, 1, 0.7) # Randomized 1:1
Y_trial <- generate_y(X_trial, Trt_trial, study_drift = 0, global_hist_drift = 0)

current_trial <- data.frame(Study = "Current", Treatment = Trt_trial, Target = Y_trial, X_trial)

# --- 6. INSPECT RESULTS ---
all_data <- do.call(rbind, c(historical_data, list(current_trial)))

# Check the Biomarker interaction in the Current Trial
# Bio_1 should show a much higher target value when Treatment == 1
library(ggplot2)
ggplot(current_trial, aes(x = factor(Bio_1), y = Target, fill = factor(Treatment))) +
  geom_boxplot() +
  labs(title = "Treatment Effect Modulated by Bio_1 (Predictive Biomarker)",
       x = "Biomarker 1 Status", fill = "Treatment") +
  theme_minimal()



# Helper: simulate one dataset (returns list with hist + current data frames)
simulate_one <- function(H = 5,
                         n_hist = 200,
                         n_trial = 500,
                         global_hist_drift = 0) {
  hist_list <- lapply(1:H, function(s) {
    X <- generate_x(n_hist)
    Trt <- rep(0, n_hist)
    Y <- generate_y(X, Trt,
                    study_drift = runif(1, -2, 2),
                    global_hist_drift = global_hist_drift)
    data.frame(Study = paste0("H", s), Treatment = Trt, Target = Y, X)
  })

  X_trial <- generate_x(n_trial)
  Trt_trial <- rbinom(n_trial, 1, 0.7)
  Y_trial <- generate_y(X_trial, Trt_trial,
                        study_drift = 0,
                        global_hist_drift = 0)
  current <- data.frame(Study = "Current", Treatment = Trt_trial, Target = Y_trial, X_trial)

  list(historical = do.call(rbind, hist_list), current = current)
}

# Compute ATE estimate and SE from an lm fitted on a formula including Treatment and Treatment:Bio_1
ate_from_lm <- function(fit, current_df) {
  coefs <- coef(fit)
  vc <- vcov(fit)
  mean_bio <- mean(current_df$Bio_1)
  # if interaction exists
  b_trt <- coefs["Treatment"]
  b_int <- if ("Treatment:Bio_1" %in% names(coefs)) coefs["Treatment:Bio_1"] else 0
  ate_hat <- b_trt + b_int * mean_bio

  # variance: Var(b_trt) + mean^2 Var(b_int) + 2 mean Cov(b_trt,b_int)
  var_trt <- vc["Treatment", "Treatment"]
  var_int <- if ("Treatment:Bio_1" %in% rownames(vc)) vc["Treatment:Bio_1", "Treatment:Bio_1"] else 0
  cov_trt_int <- if ("Treatment:Bio_1" %in% rownames(vc)) vc["Treatment", "Treatment:Bio_1"] else 0
  se_ate <- sqrt(var_trt + (mean_bio^2) * var_int + 2 * mean_bio * cov_trt_int)

  ci_lower <- ate_hat - 1.96 * se_ate
  ci_upper <- ate_hat + 1.96 * se_ate

  list(ate = ate_hat, se = se_ate, ci = c(lower = ci_lower, upper = ci_upper))
}

# Fit models and evaluate (one replication)
fit_and_eval <- function(hist_df, current_df) {
  # formula that matches data generation structure (include Bio_1 interaction)
  fm <- as.formula("Target ~ Treatment * Bio_1 + Lab_1 + Lab_2 + Lab_3 + Lab_4 + Lab_5")

  # 1) Current-only
  fit_current <- lm(fm, data = current_df)

  # 2) Stacked naive (hist + current)
  stacked <- rbind(hist_df, current_df)
  fit_stacked <- lm(fm, data = stacked)

  # 3) Stacked + study fixed effect
  fit_stack_study <- lm(update(fm, . ~ . + factor(Study)), data = stacked)

  true_ate <- mean(5 + 12 * current_df$Bio_1)

  res <- list(
    true_ate = true_ate,
    current = ate_from_lm(fit_current, current_df),
    stacked = ate_from_lm(fit_stacked, current_df),
    stacked_study = ate_from_lm(fit_stack_study, current_df)
  )
  res
}

# Top-level: run simulation grid over drifts and repetitions
run_simulation <- function(drifts = c(-3, -1, 0, 1, 3),
                           reps = 200,
                           H = 5, n_hist = 200, n_trial = 500) {
  library(dplyr)
  out <- list()

  for (d in drifts) {
    sim_res <- replicate(reps, {
      s <- simulate_one(H, n_hist, n_trial, global_hist_drift = d)
      fit_and_eval(s$historical, s$current)
    }, simplify = FALSE)

    # extract metrics
    df <- tibble::tibble(rep = 1:reps) %>%
      mutate(
        true = sapply(sim_res, function(x) x$true_ate),
        ate_current = sapply(sim_res, function(x) x$current$ate),
        se_current = sapply(sim_res, function(x) x$current$se),
        ci_l_current = sapply(sim_res, function(x) x$current$ci["lower"]),
        ci_u_current = sapply(sim_res, function(x) x$current$ci["upper"]),

        ate_stacked = sapply(sim_res, function(x) x$stacked$ate),
        se_stacked = sapply(sim_res, function(x) x$stacked$se),
        ci_l_stacked = sapply(sim_res, function(x) x$stacked$ci["lower"]),
        ci_u_stacked = sapply(sim_res, function(x) x$stacked$ci["upper"]),

        ate_study = sapply(sim_res, function(x) x$stacked_study$ate),
        se_study = sapply(sim_res, function(x) x$stacked_study$se),
        ci_l_study = sapply(sim_res, function(x) x$stacked_study$ci["lower"]),
        ci_u_study = sapply(sim_res, function(x) x$stacked_study$ci["upper"])
      )

    summarize_metrics <- function(ate, ci_l, ci_u, true) {
      bias <- mean(ate - true)
      rmse <- sqrt(mean((ate - true)^2))
      coverage <- mean(ci_l <= true & ci_u >= true)
      c(bias = bias, rmse = rmse, coverage = coverage)
    }

    metrics <- rbind(
      current = summarize_metrics(df$ate_current, df$ci_l_current, df$ci_u_current, df$true),
      stacked = summarize_metrics(df$ate_stacked, df$ci_l_stacked, df$ci_u_stacked, df$true),
      stacked_study = summarize_metrics(df$ate_study, df$ci_l_study, df$ci_u_study, df$true)
    )
    out[[as.character(d)]] <- list(metrics = metrics, raw = df)
  }

  out
}