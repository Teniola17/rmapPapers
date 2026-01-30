
set.seed(123)

# Historical trials
J <- 5
n_per_trial <- 60
N_hist <- J * n_per_trial

# True parameters
mu_alpha <- 10
tau_alpha <- 1.5
gamma_true <- 0.8
sigma_true <- 2

# Trial-level intercepts
alpha_j <- rnorm(J, mu_alpha, tau_alpha)

# Patient-level data
trial <- rep(1:J, each = n_per_trial)
x <- rnorm(N_hist)
x <- x - mean(x)  # center baseline

y <- alpha_j[trial] + gamma_true * x + rnorm(N_hist, 0, sigma_true)

hist_data <- list(
  N = N_hist,
  J = J,
  trial = trial,
  x = x,
  y = y
)


# hierarchical model to predict the new theta

library(cmdstanr)

mod_hist <- cmdstan_model("map_prior_raw2.stan")

fit_hist <- mod_hist$sample(
  data = hist_data,
  chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000,
  seed = 123
)

alpha_new_draws <- fit_hist$draws("alpha_new") |> as.numeric()

# construct MAP prior using mixture approximation


library(RBesT)

mix <- automixfit(alpha_new_draws, Nc=1:10)

weight <- mix["w",]
means <- mix["m",]
sds <- mix["s",]
K <- length(weight)

### Current trial


N_cur <- 120

alpha_true <- 10.5
beta_true <- 2.0

trt <- rbinom(N_cur, 1, 0.5)
x_cur <- rnorm(N_cur)
x_cur <- x_cur - mean(x_cur)

y_cur <- alpha_true +
  beta_true * trt +
  gamma_true * x_cur +
  rnorm(N_cur, 0, sigma_true)

cur_data <- list(
  N = N_cur,
  y = y_cur,
  trt = trt,
  x = x_cur,
  K = K,
  weights = weight,
  mu = means,
  sigma = sds
)


### fit current trial


mod_cur <- cmdstan_model("new_trial_map.stan")

fit_cur <- mod_cur$sample(
  data = cur_data,
  chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000,
  seed = 456
)

fit_cur$summary()


#### Robust version ##########################################


robus <- robustify(mix, weight=0.8, mean=0, sigma=10)

rob_w <- robus["w",]
rob_m <- robus["m",]
rob_s <- robus["s",]
k <- length(rob_w)


data_curr <- list(
  N = N_cur,
  y = y_cur,
  trt = trt,
  x = x_cur,
  K = k,
  weights = rob_w,
  mu = rob_m,
  sigma = rob_s
)

#mod_rob_cur <- cmdstan_model("rmap_curdat.stan")

fit_cur_rob <- mod_cur$sample(
  data = data_curr,
  chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000,
  seed = 456
)

fit_cur_rob$summary()



############ for MAP prior with prognostic scores

mod_ps <- cmdstan_model("borrowing_on_ps.stan")

fit_hist_ps <- mod_ps$sample(
  data = hist_data,
  chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000,
  seed = 123
)


fit_hist_ps$summary()

theta1_draw <- fit_hist_ps$draws("theta_new[1]")|>as.numeric()
theta2_draw <- fit_hist_ps$draws("theta_new[2]")|>as.numeric()

mapize1 <- automixfit(theta1_draw, Nc=1:10)

# intercept
alpha_m <- mapize1["m",]
alpha_s <- mapize1["s",]
alpha_w <- mapize1["w",]
alpha_k <- length(alpha_m)

# covariate
mapize2 <- automixfit(theta2_draw, Nc=1:10)
beta_m <- mapize2["m",]
beta_s <- mapize2["s",]
beta_w <- mapize2["w",]
beta_k <- length(beta_m)
# Apply it to the current data

mod_ps_mapcur <- cmdstan_model("current_trial_with_ps.stan")

data_curr_ps <- list(
  N = N_cur,
  y = y_cur,
  trt = trt,
  x = x_cur,
  
  # aplha components
  K_alpha = alpha_k,
  mu_alpha = alpha_m,
  sigma_alpha = alpha_s,
  w_alpha = alpha_w,
  
  # gamma componets
  K_gamma = beta_k,
  mu_gamma = beta_m,
  sigma_gamma = beta_s,
  w_gamma = beta_w
)

fit_curmap_ps <- mod_ps_mapcur$sample(
  data = data_curr_ps,
  chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000,
  seed = 123
)


fit_curmap_ps$summary()


# RMAP for progstic scores

rob_ps1 <- robustify(mapize1, weight=0.8, mean=0, sigma=10)
rob_ps2 <- robustify(mapize2, weight=0.8, mean=0, sigma=10)

alphar_m <- rob_ps1["m",]
alphar_s <- rob_ps1["s",]
alphar_w <- rob_ps1["w",]
alphar_k <- length(alphar_m)

betar_m <- rob_ps2["m",]
betar_s <- rob_ps2["s",]
betar_w <- rob_ps2["w",]
betar_k <- length(betar_m)



data_curr_ps2 <- list(
  N = N_cur,
  y = y_cur,
  trt = trt,
  x = x_cur,
  
  # aplha components
  K_alpha = alphar_k,
  mu_alpha = alphar_m,
  sigma_alpha = alphar_s,
  w_alpha = alphar_w,
  
  # gamma componets
  K_gamma = betar_k,
  mu_gamma = betar_m,
  sigma_gamma = betar_s,
  w_gamma = betar_w
)


fit_curRmap_ps <- mod_ps_mapcur$sample(
  data = data_curr_ps2,
  chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000,
  seed = 123
)


fit_curRmap_ps$summary()
