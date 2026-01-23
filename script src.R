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


#### Robust version


w_robust <- 0.5           
mu_vague <- 0
sd_vague <- 20

robust_weights <- c(
  w_robust * weight,
  1 - w_robust
)

robust_mu <- c(
  means,
  mu_vague
)

robust_sigma <- c(
  sds,
  sd_vague
)

robust_map <- list(
  K = length(robust_weights),
  weights = robust_weights,
  mu = robust_mu,
  sigma = robust_sigma
)




#######   with PS


mix_alpha <- automixfit(theta_new_draws[,1])
mix_gamma <- automixfit(theta_new_draws[,2])
