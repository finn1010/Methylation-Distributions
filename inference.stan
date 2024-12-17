functions {
    vector ss_init_prob(real mu, real gamma) {
        vector[3] probs;
        real denom = (mu + gamma)^2;

        probs[1] = gamma^2 / denom;
        probs[2] = 2 * mu * gamma / denom;
        probs[3] = mu^2 / denom;

        return probs;
    }
  matrix rate_matrix(real gamma, real mu) {
    matrix[3, 3] R;
    R[1, 1] = -2 * gamma;
    R[1, 2] = mu;
    R[1, 3] = 0;
    
    R[2, 1] = 2 * gamma;
    R[2, 2] = -(gamma + mu);
    R[2, 3] = 2 * mu;
    
    R[3, 1] = 0;
    R[3, 2] = gamma;
    R[3, 3] = -2 * mu;
    
    return R;
  }

 // vector diploid_prob(real t, vector initial_state, real gamma, real mu) {
  //  matrix[3, 3] R = rate_matrix(gamma, mu);
 //   matrix[3, 3] P = matrix_exp(R * t); // Matrix exponential for transition probabilities
 //   return P * initial_state;
 // }

  vector diploid_prob(real t, vector initial_state, real gamma, real mu) {
    matrix[3, 3] R = rate_matrix(gamma, mu);
    matrix[3, 1] initial_state_matrix;
    initial_state_matrix[, 1] = initial_state;
    matrix [3,1] F = scale_matrix_exp_multiply(t,R,initial_state_matrix);
    return to_vector(F);
    }
  vector cnLOH_event_prob(vector probabilities) {
    // Extract final diploid probabilities
    real m_prob = probabilities[1];
    real k_prob = probabilities[2];
    real w_prob = probabilities[3];

    // Update probabilities according to the cnLOH event
    m_prob += 0.5 * k_prob;
    w_prob += 0.5 * k_prob;
    k_prob = 0;

    // Return updated probabilities as a vector
    vector[3] initial_cnLOH_probs;
    initial_cnLOH_probs[1] = m_prob;
    initial_cnLOH_probs[2] = k_prob;
    initial_cnLOH_probs[3] = w_prob;

    return initial_cnLOH_probs;
  }

}

data{
    int<lower=1> K;                  //number of mixture components
    int<lower=0> N;                 //num sites
    array[N] real<lower=0,upper=1> y;    //observed beta values
    int<lower=0> age;
//ragged array for different sets of y values stan docs and carbine examples
// loop through no of patients and no of events


}
transformed data {
    ordered[3] position;
    position[1] = 0.0;
    position[2] = 0.5;
    position[3] = 1.0;
}


parameters{
            //mixture weights
    real<lower=0> kappa;                //standard deviation of peak
    real<lower=0> mu;                   //methylation rate
    real<lower=0> gamma;                //demethylation rate
    real<lower=0,upper=age> t;          //CNA time
    real<lower=0,upper=1> eta;          //Beta dist parameter
    real<lower=0,upper=1> delta;        //Beta dist parameter


}

transformed parameters{
    array[K+1] real<lower=0> a;  // Beta distribution shape parameter a
    array[K+1] real<lower=0> b;  // Beta distribution shape parameter b
    array[K+1] real pos_obs; 
    vector[K+1] cache_theta;

    for (i in 1:K+1){
        pos_obs[i] = (eta-delta) * position[i] + delta;
       // print(pos_obs[i]);
    }

    for (i in 1:K+1) {
        a[i] = kappa * pos_obs[i]; 
        b[i] = kappa * (1 - pos_obs[i]);
    }
    
    vector[K+1] initial_state = ss_init_prob(mu, gamma);
    //vector[K+1] state_probs = diploid_prob((t), initial_state, gamma, mu);
    vector[K+1] post_cnLOH_state_probs = cnLOH_event_prob(initial_state);
    vector[K+1] altered_state_probs = diploid_prob((age-t), post_cnLOH_state_probs, gamma, mu);
    cache_theta = altered_state_probs;
    if (abs(sum(cache_theta)-1) > 1e-3){
        reject("abs confirmed mu: ", mu, " gamma: ", gamma, " t: ", t);
    }
}

model {
    // Priors
    // theta and CNA time are flat priors
    kappa ~ lognormal(3.8, 0.5);       
    delta ~ beta(5,95);
    eta ~ beta(95,5);
    mu ~ normal(0,0.05);
    gamma ~ normal(0,0.05);


    // Likelihood
    for (n in 1:N) {
        vector[K+1] log_likelihood = log(cache_theta);
        
        for (k in 1:K+1) {
            log_likelihood[k] += beta_lpdf(y[n] | a[k], b[k]); // Beta log-density
        }
        target += log_sum_exp(log_likelihood); // Log mixture likelihood
    }
}

