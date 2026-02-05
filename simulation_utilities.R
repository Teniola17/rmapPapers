library(MASS)

# --- 1. SETUP PARAMETERS ---
n_hist <- 200      # Patients per historical study
n_trial <- 500     # Patients in current trial
H <- 5             # Number of historical studies
p_cont <- 50       # Number of continuous features
p_bin <- 5         # Number of binary biomarkers
rho <- 0.8         # Correlation for collinearity

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
generate_y <- function(X, Trt, study_drift = 0) {
  # 3a. Prognostic (Baseline) Signal
  # Linear + Non-linear + Interaction between features
  prog_signal <- (2 * X[, "Lab_1"]) +
    (3 * X[, "Lab_2"]^2) +
    (5 * sin(X[, "Lab_3"])) +
    (4 * X[, "Lab_4"] * X[, "Lab_5"])

  # 3b. Predictive (Treatment) Signal
  # Treatment has a base effect + specific interaction with Bio_1
  trt_effect <- (5 * Trt) + (12 * Trt * X[, "Bio_1"])

  # 3c. Noise and Study Drift
  noise <- rnorm(nrow(X), mean = 0, sd = 2)

  y <- prog_signal + trt_effect + study_drift + noise
  return(y)
}


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
  Y <- generate_y(X, Trt, study_drift = runif(1, -2, 2)) # Slight variation between studies
  
  data.frame(Study = paste0(s), Treatment = Trt, Target = Y, X)
})

# --- 5. GENERATE CURRENT TRIAL DATA (Placebo + Treatment) ---
X_trial <- generate_x(n_trial)
Trt_trial <- rbinom(n_trial, 1, 0.7) # Randomized 1:1
Y_trial <- generate_y(X_trial, Trt_trial, study_drift = 0)

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

