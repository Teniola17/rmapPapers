library(cmdstanr)
library(posterior)

set.seed(2026)

# ----------------------------
# Stan model (vector y0 version)
# ----------------------------
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
mod <- cmdstan_model(stan_file)

# ----------------------------
# Helper: simulate + fit one scenario
# ----------------------------
run_scenario <- function(mu0_true,
                         muC_true = 0,
                         muT_true = 0.4,
                         sigma_true = 1,
                         n0 = 80, nC = 60, nT = 60,
                         a = 1,
                         alpha_shape = 1, beta_shape = 1,
                         seed = 2026) {
  
  set.seed(seed)
  
  y0 <- rnorm(n0, mean = mu0_true, sd = sigma_true)
  yC <- rnorm(nC, mean = muC_true, sd = sigma_true)
  yT <- rnorm(nT, mean = muT_true, sd = sigma_true)
  
  # constraint for propriety of C(delta):
  # 0.5*delta*n0 + a - 1.5 > 0  => delta > (3 - 2a)/n0
  eps <- max(1e-6, (3 - 2*a)/n0 + 1e-6)
  eps <- min(eps, 0.999999)
  
  data_list <- list(
    nT = nT, yT = yT,
    nC = nC, yC = yC,
    n0 = n0, y0 = y0,
    alpha_shape = alpha_shape,
    beta_shape  = beta_shape,
    a = a,
    eps = eps
  )
  
  fit <- mod$sample(
    data = data_list,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    seed = seed,
    refresh = 200
  )
  
  draws <- fit$draws(c("Delta", "delta", "ess"), format = "draws_df")
  
  list(
    fit = fit,
    draws = draws,
    truth = list(Delta_true = muT_true - muC_true, mu0_true = mu0_true),
    data = list(y0 = y0, yC = yC, yT = yT)
  )
}

# ----------------------------
# Three drift scenarios
# ----------------------------
# Drift is difference between historical control mean (mu0_true) and current control mean (muC_true=0)
scenarios <- list(
  "No drift (mu0 = 0.00)"       = 0.00,
  "Moderate drift (mu0 = 0.20)" = 0.20,
  "Strong drift (mu0 = 0.50)"   = 0.50
)

res <- lapply(seq_along(scenarios), function(i) {
  nm <- names(scenarios)[i]
  mu0 <- scenarios[[i]]
  cat("\n--- Fitting:", nm, " ---\n")
  out <- run_scenario(mu0_true = mu0, seed = 2026 + i)
  out$name <- nm
  out
})
names(res) <- names(scenarios)

# ----------------------------
# Summary table
# ----------------------------
sum_one <- function(draws) {
  s <- summarise_draws(as_draws_df(draws))
  # make it easy to read
  s[, c("variable", "mean", "median", "sd", "q5", "q95", "rhat", "ess_bulk")]
}

summaries <- lapply(res, function(x) sum_one(x$draws))
for (nm in names(summaries)) {
  cat("\n==============================\n", nm, "\n==============================\n")
  print(summaries[[nm]])
}

# ----------------------------
# Overlay plots: Delta and delta
# ----------------------------
par(mfrow = c(1, 1))

# Delta densities
plot(NULL, xlim = c(-0.4, 1.2), ylim = c(0, 4),
     xlab = "Delta", ylab = "Density", main = "Posterior of Delta")
cols <- c("black", "blue", "red")
i <- 1
for (nm in names(res)) {
  d <- density(res[[nm]]$draws$Delta)
  lines(d$x, d$y, col = cols[i], lwd = 2)
  i <- i + 1
}
abline(v = res[[1]]$truth$Delta_true, lwd = 2, lty = 2)
legend("topleft", legend = names(res), col = cols, lwd = 2, bty = "n")
mtext("Dashed line = true Delta", side = 3, line = 0.2, cex = 0.85)

# delta densities
plot(NULL, xlim = c(0, 1), ylim = c(0, 6),
     xlab = "delta", ylab = "Density", main = "Posterior of delta (borrowing)")
i <- 1
for (nm in names(res)) {
  d <- density(res[[nm]]$draws$delta)
  lines(d$x, d$y, col = cols[i], lwd = 2)
  i <- i + 1
}
legend("topleft", legend = names(res), col = cols, lwd = 2, bty = "n")

# ----------------------------
# A simple numeric comparison: posterior P(Delta > 0)
# ----------------------------
prob_tbl <- data.frame(
  scenario = names(res),
  P_Delta_gt_0 = sapply(res, function(x) mean(x$draws$Delta > 0)),
  mean_Delta = sapply(res, function(x) mean(x$draws$Delta)),
  mean_delta = sapply(res, function(x) mean(x$draws$delta)),
  mean_ess   = sapply(res, function(x) mean(x$draws$ess))
)
cat("\n\nPosterior comparisons:\n")
print(prob_tbl, row.names = FALSE)