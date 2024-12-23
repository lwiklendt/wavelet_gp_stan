// Templates for generating Stan code.
// The blocks of code enclosed between /*** start ... ***/ and /*** end ... ***/ are to be
// replicated for each independent group/random-effect term, where
// {v} is the grouping term label, which can just be b1, b2, b3, ..., etc,
// {ncol} is the number of columns in the grouping term's design matrix,
// {nlev} is the number of independent levels in the grouping term.

data {
  int N;  // num rows or observations
  int P;  // num mu       population effects
  int Q;  // num residual population effects
  int F;  // num frequencies
  
  matrix[N,P] X;  // mu       population predictors
  matrix[N,Q] W;  // residual population predictors
  
  matrix[N,F] y;  // observations

  matrix[F,F] kern_chol;  // cholesky decomposed kernel matrix
  
  /*** start data onecol ***/
  
  // {v} (1|{nlev}): {term}
  vector[N] Z_{v};
  array[N] int<lower=1, upper={nlev}> l_{v};
  
  /*** end data onecol ***/
  /*** start data multicol ***/
  
  // {v} ({ncol}|{nlev}): {term}
  matrix[N,{ncol}] Z_{v};
  array[N] int<lower=1, upper={nlev}> l_{v};
  
  /*** end data multicol ***/
}


transformed data {
  real mean_y = mean(y);
}


parameters {
  matrix[F,P] z_beta;
  matrix[F,Q] z_gamma;
  
  real offset_eta;
  real<lower=0> tau_beta;
  real<lower=0> tau_gamma;

  /*** start parameters onecol ***/
  
  real<lower=0> sigma_{v};
  matrix[F,{nlev}] z_{v};
  
  /*** end parameters onecol ***/
  /*** start parameters multicol ***/
  
  vector<lower=0>[{ncol}] sigma_{v};
  cholesky_factor_corr[{ncol}] chol_corr_{v};
  matrix[F,{ncol}] z_{v}[{nlev}];
  
  /*** end parameters multicol ***/
}


transformed parameters {
  
  matrix[P,F] beta;
  matrix[Q,F] gamma;
  
  /*** start transformed parameter declarations onecol ***/
  
  matrix[{nlev},F] {v};
  
  /*** end transformed parameter declarations onecol ***/
  /*** start transformed parameter declarations multicol ***/
  
  matrix[{ncol},F] {v}[{nlev}];
  
  /*** end transformed parameter declarations multicol ***/
  
  {
    for (q in 1:Q) {
      gamma[q,] = tau_gamma * (kern_chol * z_gamma[,q])';
    }
    
    for (p in 1:P) {
      beta[p,] = tau_beta * (kern_chol * z_beta[,p])';
    }
    
    /*** start transformed parameter definitions onecol ***/
    
    for (l in 1:{nlev}) {{
      {v}[l] = sigma_{v} * (kern_chol * z_{v}[,l])';
    }}

    /*** end transformed parameter definitions onecol ***/
    /*** start transformed parameter definitions multicol ***/
    
    // sample each column
    for (c in 1:{ncol}) {{
      for (l in 1:{nlev}) {{
        {v}[l][c,] = sigma_{v}[c] * (kern_chol * z_{v}[l][,c])';
      }}
    }}
    
    // correlate columns
    for (l in 1:{nlev}) {{
        {v}[l] = chol_corr_{v} * {v}[l];
    }}
        
    /*** end transformed parameter definitions multicol ***/
  }
}


model {
  matrix[N,F] eta;
  matrix[N,F] log_omega;
  
  offset_eta ~ student_t(3, mean_y, 4);
  tau_gamma ~ prior_tau_gamma;
  tau_beta  ~ prior_tau_beta;

  eta = X * beta + offset_eta;
  log_omega = W * gamma;
  
  to_vector(z_beta)  ~ normal(0, 1);
  to_vector(z_gamma) ~ normal(0, 1);
  
  /*** start model onecol ***/
  
  sigma_{v} ~ prior_sigma_{v};
  to_vector(z_{v}) ~ normal(0, 1);
  for (n in 1:N) {{
    int l = l_{v}[n];
    {dpar}[n,] += Z_{v}[n] * {v}[l];
  }}
  
  /*** end model onecol ***/
  /*** start model multicol ***/
  
  sigma_{v} ~ prior_sigma_{v};
  chol_corr_{v} ~ lkj_corr_cholesky(2);
  for (n in 1:N) {{
    int l = l_{v}[n];
    to_vector(z_{v}[l]) ~ normal(0, 1);
    {dpar}[n,] += Z_{v}[n,] * {v}[l];
  }}
  
  /*** end model multicol ***/
  
  // calculate residuals
  {
    to_vector(y) ~ normal(to_vector(eta), exp(to_vector(log_omega)));
  }
}

generated quantities {
  
  /*** start generated quantities declarations onecol ***/
  
  vector[F] new_{v};
  
  /*** end generated quantities declarations onecol ***/
  /*** start generated quantities declarations multicol ***/
  
  matrix[{ncol},F] new_{v};
  
  /*** end generated quantities declarations multicol ***/
  {
    vector[F] new_z;  // for sampling from standard normal
    
    /*** start generated quantities definitions onecol ***/
    
    for (fi in 1:F) {{
      new_z[fi] = normal_rng(0, 1);
    }}
    new_{v} = sigma_{v} * (kern_chol * new_z);
    
    /*** end generated quantities definitions onecol ***/
    /*** start generated quantities definitions multicol ***/
    
    // sample each column for new level
    for (c in 1:{ncol}) {{
      for (fi in 1:F) {{
        new_z[fi] = normal_rng(0, 1);
      }}
      new_{v}[c,] = sigma_{v}[c] * (kern_chol * new_z)';
    }}
    
    // correlate columns
    new_{v} = chol_corr_{v} * new_{v};
      
    /*** end generated quantities definitions multicol ***/
  }
}
