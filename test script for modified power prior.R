# ---- setup ----
library(cmdstanr)
library(posterior)

set.seed(2026)

# ---- simulate a simple example ----
n0 <- 80   # historical control
nC <- 60   # current control
nT <- 60   # current treatment

muC_true <- 0.0
muT_true <- 0.4              # true treatment benefit (Delta = 0.4)
sigma_true <- 1.0

# add a bit of drift: historical control slightly different
mu0_true <- 0.15             # drift in historical control mean

y0 <- rnorm(n0, mean = mu0_true, sd = sigma_true)
yC <- rnorm(nC, mean = muC_true, sd = sigma_true)
yT <- rnorm(nT, mean = muT_true, sd = sigma_true)

# ---- priors ----
a <- 1.0                     # Jeffreys-like exponent: p0 ∝ (1/sigma^2)^a
alpha_shape <- 1.0           # Beta prior on delta
beta_shape  <- 1.0           # uniform on (0,1)

# eps must ensure: 0.5*delta*n0 + a - 1.5 > 0  => delta > (3 - 2a)/n0
eps <- max(1e-6, (3 - 2*a)/n0 + 1e-6)
eps <- min(eps, 0.999999)

# ---- write Stan model to a file ----
stan_code <- '
functions {
  real log_C_delta_vector(real delta, vector y0, real a) {
    int n0 = rows(y0);
    real y0bar = mean(y0);
    real S0 = dot_self(y0 - y0bar);

    real m = delta * n0;
    real shape = 0.5 * m + a - 1.5;

    return
        (-0.5 * m) * log(2*pi())
      + 0.5 * (log(2*pi()) - log(m))
      + lgamma(shape)
      + shape * (log(2.0) - log(delta) - log(S0));
  }
}

data {
  int<lower=1> nT;
  vector[nT] yT;

  int<lower=1> nC;
  vector[nC] yC;

  int<lower=2> n0;
  vector[n0] y0;

  real<lower=0> alpha_shape;
  real<lower=0> beta_shape;

  real<lower=0> a;
  real<lower=0, upper=1> eps;
}

parameters {
  real muT;
  real muC;
  real<lower=0> sigma;
  real<lower=eps, upper=1> delta;
}

model {
  // prior for delta (truncated by parameter bounds)
  delta ~ beta(alpha_shape, beta_shape);

  // baseline prior: p0(muT, muC, sigma^2) ∝ (1/sigma^2)^a
  target += -2.0 * a * log(sigma);

  // current data likelihood
  yT ~ normal(muT, sigma);
  yC ~ normal(muC, sigma);

  // powered historical likelihood
  target += delta * normal_lpdf(y0 | muC, sigma);

  // modified power prior normalization
  {
    real m = delta * n0;
    real shape = 0.5 * m + a - 1.5;
    if (shape <= 0)
      reject("Invalid delta: need 0.5*delta*n0 + a - 1.5 > 0");
  }
  target += -log_C_delta_vector(delta, y0, a);
}

generated quantities {
  real Delta = muT - muC;
  real ess = delta * n0;
}
'

stan_file <- write_stan_file(stan_code)

# ---- compile ----
mod <- cmdstan_model(stan_file)

# ---- data list for Stan ----
data_list <- list(
  nT = nT, yT = yT,
  nC = nC, yC = yC,
  n0 = n0, y0 = y0,
  alpha_shape = alpha_shape,
  beta_shape  = beta_shape,
  a = a,
  eps = eps
)

# ---- sample ----
fit <- mod$sample(
  data = data_list,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  seed = 2026
)

print(fit$summary(variables = c("Delta", "delta", "ess", "muT", "muC", "sigma")))

# ---- quick posterior summaries ----
draws <- fit$draws(c("Delta", "delta", "ess"), format = "draws_df")

posterior::summarise_draws(draws)

# ---- quick plots (base R) ----
par(mfrow = c(1, 2))
hist(draws$Delta, breaks = 40, main = "Posterior of Delta", xlab = "Delta")
abline(v = muT_true - muC_true, lwd = 2)

hist(draws$delta, breaks = 40, main = "Posterior of delta", xlab = "delta")