// =============================================================================
// ar1ns_predictive_prior.stan
//
// PURPOSE
//   Sample from the posterior predictive prior for the current-trial parameter
//   theta_0, integrating over (theta_H, phi) given K historical sufficient
//   statistics under the NON-STATIONARY AR(1) prior.
//
// MODEL
//   Likelihood  : ybar_k | theta_k  ~ N(theta_k, s2_k),   k = 1,...,K
//   AR(1) prior : theta_1           ~ N(mu1, sig1sq)
//                 theta_k | theta_{k-1} ~ N(mu + rho^delta_k * (theta_{k-1}-mu),
//                                           tau2*(1-rho^{2*delta_k})/(1-rho^2))
//   Equivalent joint:
//                 theta_H | phi ~ N(mu_H, Sigma_ns)
//   where:
//     mu_H[k]       = mu + (mu1 - mu) * rho^d_k          (d_k = t_k - t_1)
//     Sigma_ns[j,k] = (tau2/(1-rho^2)) * rho^|d_j-d_k|   (stationary part)
//                   + xi * rho^d_j * rho^d_k              (non-stationary rank-1 correction)
//     xi            = sig1sq - tau2/(1-rho^2)             (non-stationarity excess)
//
// GENERATED QUANTITIES
//   theta_0 is drawn from N(mu0_cond, omega2) where:
//     mu0_cond = [mu + rho^d0*(mu1-mu)] + c' Sigma_ns^{-1} (theta_H - mu_H)
//     omega2   = sigma0sq - c' Sigma_ns^{-1} c
//     c[k]     = (tau2/(1-rho^2))*rho^(d0-d_k) + xi*rho^(d0+d_k)
//     sigma0sq = tau2/(1-rho^2) + xi*rho^(2*d0)
//
// OUTPUT
//   theta0 samples => pass to RBesT::automixfit() for mixture approximation
// =============================================================================

data {
  // ── Historical trial information ──────────────────────────────────────────
  int<lower=1>          K;         // number of historical trials
  vector[K]             ybar;      // observed sample means ybar_1,...,ybar_K
  vector<lower=0>[K]    s2;        // sampling variances s2_k = sigma^2 / n_k
  vector[K]             t_hist;    // trial times t_1 < t_2 < ... < t_K

  // ── Current trial time ────────────────────────────────────────────────────
  real                  t0;        // t0 > t_K

  // ── Hyperprior specifications ─────────────────────────────────────────────
  // mu1  ~ N(mu1_m, mu1_s^2)       initial mean
  real                  mu1_m;
  real<lower=0>         mu1_s;

  // sig1sq ~ InvGamma(a1, b1)      initial variance (non-stationary)
  real<lower=0>         sig1sq_a;
  real<lower=0>         sig1sq_b;

  // mu ~ N(mu_m, mu_s^2)           long-run mean
  real                  mu_m;
  real<lower=0>         mu_s;

  // tau2 ~ InvGamma(atau, btau)    innovation variance
  real<lower=0>         tau2_a;
  real<lower=0>         tau2_b;

  // rho ~ Beta(a_rho, b_rho)       autocorrelation  (constrained to (0,1))
  // Use a_rho > b_rho to favour positive autocorrelation, e.g. (2, 1)
  real<lower=0>         rho_a;
  real<lower=0>         rho_b;

  // ── Numerical jitter for PD stability ────────────────────────────────────
  real<lower=0>         jitter;    // recommend 1e-6
}

// ── Pre-compute elapsed times (fixed across all iterations) ─────────────────
transformed data {
  vector[K] d;       // d[k] = t_k - t_1  (d[1] = 0 by construction)
  real      d0;      // d0   = t0  - t_1

  for (k in 1:K) d[k] = t_hist[k] - t_hist[1];
  d0 = t0 - t_hist[1];
}

// ── Unknown quantities ────────────────────────────────────────────────────────
parameters {
  real              mu1;          // initial mean
  real<lower=0>     sig1sq;       // initial variance
  real              mu;           // long-run mean
  real<lower=0,
       upper=1>     rho;          // AR coefficient in (0,1) for real-power stability
  real<lower=0>     tau2;         // innovation variance
  vector[K]         theta_hist;   // historical trial parameters
}

// ── Derived quantities used in both model and generated quantities ────────────
transformed parameters {
  // Stationary variance and non-stationarity excess
  real         stat_var  = tau2 / (1.0 - rho^2);
  real         xi        = sig1sq - stat_var;

  // Initial-decay vector: decay[k] = rho^d[k]
  vector[K]    decay;
  for (k in 1:K) decay[k] = pow(rho, d[k]);

  // Non-stationary prior mean vector
  vector[K]    mu_H;
  for (k in 1:K) mu_H[k] = mu + (mu1 - mu) * decay[k];

  // Non-stationary covariance matrix Sigma_ns = Sigma_st + xi * decay * decay'
  // Build element-wise then Cholesky-decompose for efficiency
  matrix[K, K] Sigma_ns;
  for (j in 1:K)
    for (k in 1:K)
      Sigma_ns[j, k] = stat_var * pow(rho, fabs(d[j] - d[k]))
                     + xi * decay[j] * decay[k];

  // Add diagonal jitter to ensure positive definiteness
  matrix[K, K] Sigma_ns_jit = add_diag(Sigma_ns, jitter);

  // Cholesky factor  (used in the likelihood and for solves)
  matrix[K, K] L = cholesky_decompose(Sigma_ns_jit);
}

// ── Joint log-posterior ────────────────────────────────────────────────────────
model {
  // ── Hyperpriors ──────────────────────────────────────────────────────────
  mu1    ~ normal(mu1_m, mu1_s);
  sig1sq ~ inv_gamma(sig1sq_a, sig1sq_b);
  mu     ~ normal(mu_m, mu_s);
  tau2   ~ inv_gamma(tau2_a, tau2_b);
  rho    ~ beta(rho_a, rho_b);

  // ── Non-stationary AR(1) prior on theta_hist ─────────────────────────────
  theta_hist ~ multi_normal_cholesky(mu_H, L);

  // ── Gaussian likelihood from historical sufficient statistics ─────────────
  // ybar_k | theta_k ~ N(theta_k, s2_k)
  ybar ~ normal(theta_hist, sqrt(s2));
}

// ── Draw theta_0 from its predictive conditional distribution ─────────────────
generated quantities {
  real theta0;

  {
    // Cross-covariance vector  c[k] = Cov(theta_k, theta_0 | phi)
    // = stat_var * rho^(d0-d[k])  +  xi * rho^(d0+d[k])
    vector[K] c_vec;
    for (k in 1:K)
      c_vec[k] = stat_var * pow(rho, d0 - d[k])
               + xi       * pow(rho, d0 + d[k]);

    // Marginal prior variance of theta_0
    real sigma0sq = stat_var + xi * pow(rho, 2.0 * d0);

    // Solve Sigma_ns \ c_vec using stored Cholesky factor L
    //   L * L' * x = c_vec  =>  x = L'^{-1} L^{-1} c_vec
    vector[K] Linv_c  = mdivide_left_tri_low(L, c_vec);
    vector[K] Sinv_c  = mdivide_right_tri_low(Linv_c', L)';  // = Sigma_ns^{-1} c_vec

    // Conditional variance: omega^2 = sigma0sq - c' Sigma_ns^{-1} c
    real omega2 = sigma0sq - dot_product(c_vec, Sinv_c);
    // Guard against tiny negative values from floating-point error
    real omega2_safe = fmax(omega2, 1e-10);

    // Conditional mean:
    //   mu0_cond = [mu + rho^d0*(mu1-mu)] + c' Sigma_ns^{-1} (theta_hist - mu_H)
    real base_mean = mu + pow(rho, d0) * (mu1 - mu);
    real mu0_cond  = base_mean
                   + dot_product(Sinv_c, theta_hist - mu_H);

    // Draw theta_0 ~ N(mu0_cond, omega2)
    theta0 = normal_rng(mu0_cond, sqrt(omega2_safe));
  }
}
