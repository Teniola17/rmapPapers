# simulation 

library(MASS)
library(tidyverse)

n_hist <- 200      # Patients per historical study
n_trial <- 500     # Patients in current trial
H <- 5             # Number of historical studies
p_cont <- 50       # Number of continuous features
p_bin <- 5         # Number of binary biomarkers
rho <- 0.8         # Correlation for collinearity
global_hist_drift <- 0  # Global historical vs current shift

set.seed(123)

generate_x <- function(n) {
  sigma <- toeplitz(rho^(0:(p_cont - 1)))
  X_cont <- mvrnorm(n, mu = rep(0, p_cont), Sigma = sigma)
  X_bin <- matrix(rbinom(n * p_bin, 1, 0.3), nrow = n)
  
  colnames(X_cont) <- paste0("Lab_", 1:p_cont)
  colnames(X_bin) <- paste0("Bio_", 1:p_bin)
  
  cbind(X_cont, X_bin)
}


generate_y <- function(X, Trt,
                       study_drift = 0,
                       global_hist_drift = 0) {
  
  prog_signal <- (2 * X[, "Lab_1"]) +
    (3 * X[, "Lab_2"]^2) +
    (5 * sin(X[, "Lab_3"])) +
    (4 * X[, "Lab_4"] * X[, "Lab_5"])
  
  trt_effect <- (5 * Trt) + (12 * Trt * X[, "Bio_1"])
  
  noise <- rnorm(nrow(X), mean = 0, sd = 2)
  
  prog_signal +
    trt_effect +
    study_drift +
    global_hist_drift +
    noise
}

# for test then pool

library(metafor)

test_then_pool_meta <- function(df, alpha = 0.05) {
  
  # 1️⃣ Historical controls only
  hist_controls <- subset(df, Treatment == 0 & Study != "Current")
  
  studies <- unique(hist_controls$Study)
  
  est <- c()
  se <- c()
  
  # 2️⃣ Fit model in each historical study
  for (s in studies) {
    d <- subset(hist_controls, Study == s)
    fit <- lm(Target ~ X, data = d)
    
    # adjusted mean = intercept + beta * mean(X)
    beta <- coef(fit)
    vc <- vcov(fit)
    
    mean_x <- mean(d$X)
    
    mu_hat <- beta[1] + beta[2] * mean_x
    
    var_mu <- vc[1,1] + mean_x^2 * vc[2,2] + 
      2 * mean_x * vc[1,2]
    
    est <- c(est, mu_hat)
    se  <- c(se, sqrt(var_mu))
  }
  
  # 3️⃣ Random-effects meta-analysis
  meta_fit <- rma(yi = est, sei = se, method = "REML")
  
  mu_hist <- meta_fit$b[1]
  var_hist <- meta_fit$se^2
  
  # 4️⃣ Current control estimate
  curr_ctrl <- subset(df, Study == "Current" & Treatment == 0)
  fit_curr <- lm(Target ~ X, data = curr_ctrl)
  
  beta_c <- coef(fit_curr)
  vc_c <- vcov(fit_curr)
  mean_x_c <- mean(curr_ctrl$X)
  
  mu_curr <- beta_c[1] + beta_c[2] * mean_x_c
  
  var_curr <- vc_c[1,1] + mean_x_c^2 * vc_c[2,2] + 
    2 * mean_x_c * vc_c[1,2]
  
  # 5️⃣ Wald test
  z <- (mu_hist - mu_curr) / sqrt(var_hist + var_curr)
  p_val <- 2 * (1 - pnorm(abs(z)))
  
  pool_decision <- p_val > alpha
  
  list(
    p_value = p_val,
    pooled = pool_decision,
    meta_fit = meta_fit
  )
}


# generate historical data

historical_data <- lapply(1:H, function(s) {
  X <- generate_x(n_hist)
  Trt <- rep(0, n_hist)
  
  Y <- generate_y(
    X, Trt,
    study_drift = runif(1, -2, 2),
    global_hist_drift = global_hist_drift
  )
  
  data.frame(
    Study = paste0("H", s),
    Treatment = Trt,
    Target = Y,
    X
  )
})

# current trial

X_trial <- generate_x(n_trial)
Trt_trial <- rbinom(n_trial, 1, 0.7)
Y_trial <- generate_y(X_trial, Trt_trial)

current_trial <- data.frame(
  Study = "Current",
  Treatment = Trt_trial,
  Target = Y_trial,
  X_trial
)

# combined data

all_data <- do.call(rbind, c(historical_data, list(current_trial)))


# cinstruct the prognostic scores

library(tidymodels)

hist_df <- all_data %>% 
  filter(Study!="Current") %>% 
  dplyr::select(-Treatment)

# split into test and training sets

Split <- initial_split(hist_df, prop = 0.75, strata = Study )

train_df <- training(Split)
test_df <- testing(Split)

# v fold CV

folds <- vfold_cv(train_df, v = 5, strata = Study)

# recipe

rec <- recipe(Target ~ ., data = train_df) %>%
  step_zv(all_predictors()) %>%      # remove zero-variance columns
  step_normalize(all_numeric_predictors()) %>% 
  step_novel(Study) %>%
  step_dummy(Study)


# model specification

rf_model <- rand_forest(
  trees = 500,
  mtry = tune(),
  min_n = tune()
) %>%
  set_engine("ranger") %>%
  set_mode("regression")

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_model)

rf_grid <- grid_regular(
  mtry(range = c(5, 30)),
  min_n(range = c(2, 20)),
  levels = 5
)



tuned_results <- tune_grid(
  wf,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(rmse, rsq)
)


best_params <- select_best(tuned_results, metric = "rmse")

best_params


final_wf <- finalize_workflow(wf, best_params)

# fit 

final_fit <- fit(final_wf, data = train_df)

# Evaluating model

test_preds <- predict(final_fit, test_df) %>%
  bind_cols(test_df)

metrics(test_preds, truth = Target, estimate = .pred)

#### prediction

current_ml <- current_trial %>%
  dplyr::select(-Treatment)

hist_pred <- predict(final_fit, hist_df)

current_preds <- predict(final_fit, current_ml)

# combine the dataset for analysis

all_preds <- hist_pred %>% bind_rows(current_preds)

adam <- tibble(Study=all_data$Study,
               Treatment=all_data$Treatment,
               Target=all_data$Target,
               X=scale(all_preds$.pred, scale=FALSE))

## baseline model, NO PS

modBas <- adam %>% 
  filter(Study=="Current") %>% 
  lm(Target~Treatment, data=.)
  
# separate with PS

mod1 <- adam %>% 
  filter(Study=="Current") %>% 
  lm(Target~Treatment+X, data=.)


# pooling

modPooling <- adam %>% 
  lm(Target~Treatment+X, data=.)


# single arm

modSingleArm <- adam %>% 
  filter(
    Study != "Current" |  # all historical studies
      (Study == "Current" & Treatment == 1)  # only current treatment arm
  )%>% 
  lm(Target~Treatment+X, data=.)

# test then pool


pool_dec <- test_then_pool_meta(adam, 0.2)$pooled


if(pool_dec){
  TTP_df <- adam
} else{
  TTP_df <- adam %>% 
    filter(Study=="Current")
}

modTTP <- TTP_df %>% 
  lm(Target~Treatment+X, data=.)


# power prior

