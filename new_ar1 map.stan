// =============================================================================
// new_trial_posterior.stan
//
// PURPOSE
//   Compute the posterior of theta_0 for the new (current) single-arm trial,
//   using the conjugate mixture-of-normals prior obtained from
//   RBesT::automixfit() applied to theta_0 samples from the first Stan model.
//
// MODEL
//   Prior   : theta_0 ~ sum_{c=1}^{C} w_c * N(m_c, sd_c^2)   (mixture of normals)
//   Likelihood (sufficient statistic form):
//             ybar0 | theta_0 ~ N(theta_0, s0sq)
//             where s0sq = sigma^2 / n0  (known or estimated externally)
//
// NOTES
//   - The mixture prior is a weighted sum of normal densities; in log scale this
//     is a log_sum_exp, making the target non-conjugate but low-dimensional and
//     easily handled by Stan's HMC.
//   - A single continuous parameter (theta_0) means sampling is very fast.
//   - For robustness, a "robust mixture" can be requested (see data input
//     `robust_weight`): a small weight is placed on a wide non-informative
//     component N(0, 1e4) following Schmidli et al. (2014).
//
// OUTPUTS
//   Posterior samples of theta_0: summarise with mean, sd, credible intervals.
// =============================================================================

data {
  // ── Current trial sufficient statistics ───────────────────────────────────
  real          ybar0;     // current trial sample mean
  real<lower=0> s0sq;      // sampling variance sigma^2 / n0

  // ── Mixture prior from RBesT::automixfit() ────────────────────────────────
  int<lower=1>     C;           // number of mixture components
  simplex[C]       w;           // mixture weights  (sum to 1)
  vector[C]        mix_means;   // component means  m_1,...,m_C
  vector<lower=0>[C] mix_sds;   // component SDs    sd_1,...,sd_C

  // ── Robust mixture option (Schmidli et al. 2014) ──────────────────────────
  // Set robust_weight = 0 to use the mixture prior unchanged.
  // Set robust_weight in (0, 0.2] to add a vague N(0, vague_sd^2) component
  // with weight robust_weight, down-weighting all other components proportionally.
  real<lower=0, upper=1> robust_weight;   // recommend 0 or 0.1
  real<lower=0>          vague_sd;        // SD of vague component, e.g. 1e2
}

// ── Validate and assemble robust weights ─────────────────────────────────────
transformed data {
  int    C_rob    = C + 1;                // augmented component count
  vector[C_rob] w_rob;                   // robust mixture weights
  vector[C_rob] m_rob;                   // augmented means
  vector<lower=0>[C_rob] s_rob;          // augmented SDs

  // Informative components get weight (1 - robust_weight) * w[c]
  for (c in 1:C) {
    w_rob[c] = (1.0 - robust_weight) * w[c];
    m_rob[c] = mix_means[c];
    s_rob[c] = mix_sds[c];
  }
  // Vague component
  w_rob[C_rob] = robust_weight;
  m_rob[C_rob] = 0.0;
  s_rob[C_rob] = vague_sd;
}

// ── Single parameter: current trial mean ─────────────────────────────────────
parameters {
  real theta0;
}

// ── Log-posterior ─────────────────────────────────────────────────────────────
model {
  // ── Mixture-of-normals prior ─────────────────────────────────────────────
  // log pi(theta0) = log sum_c w_c * N(theta0 | m_c, sd_c^2)
  //               = log_sum_exp_c [ log(w_c) + normal_lpdf(theta0 | m_c, sd_c) ]
  vector[C_rob] log_contrib;
  for (c in 1:C_rob)
    log_contrib[c] = log(w_rob[c])
                   + normal_lpdf(theta0 | m_rob[c], s_rob[c]);
  target += log_sum_exp(log_contrib);

  // ── Gaussian likelihood from current trial sufficient statistic ───────────
  // ybar0 | theta0 ~ N(theta0, s0sq)
  ybar0 ~ normal(theta0, sqrt(s0sq));
}


