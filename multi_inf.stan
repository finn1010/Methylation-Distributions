functions {
    vector ss_init_prob(real mu, real gamma) {
        vector[3] probs;
        real denom = (mu + gamma)^2;

        probs[1] = gamma^2 / denom;
        probs[2] = 2 * mu * gamma / denom;
        probs[3] = mu^2 / denom;

        return probs;
    }

    matrix dip_rate_matrix(real gamma, real mu) {
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
    
    matrix tri_rate_matrix(real mu, real gamma) {
        matrix[4, 4] R;
        R[1, 1] = -3 * mu;
        R[1, 2] = gamma;
        R[1, 3] = 0;
        R[1, 4] = 0;

        R[2, 1] = 3 * mu;
        R[2, 2] = -(gamma + 2 * mu);
        R[2, 3] = 2 * gamma;
        R[2, 4] = 0;

        R[3, 1] = 0;
        R[3, 2] = 2 * mu;
        R[3, 3] = -(2 * gamma + mu);
        R[3, 4] = 3 * gamma;

        R[4, 1] = 0;
        R[4, 2] = 0;
        R[4, 3] = mu;
        R[4, 4] = -3 * gamma;
        
        return R;
    }

    matrix tet_rate_matrix(real mu, real gamma) {
        matrix[5, 5] R;

        R[1, 1] = -4 * mu;
        R[1, 2] = gamma;
        R[1, 3] = 0;
        R[1, 4] = 0;
        R[1, 5] = 0;

        R[2, 1] = 4 * mu;
        R[2, 2] = -(3 * mu + gamma);
        R[2, 3] = 2 * gamma;
        R[2, 4] = 0;
        R[2, 5] = 0;

        R[3, 1] = 0;
        R[3, 2] = 3 * mu;
        R[3, 3] = -(2 * mu + 2 * gamma);
        R[3, 4] = 3 * gamma;
        R[3, 5] = 0;

        R[4, 1] = 0;
        R[4, 2] = 0;
        R[4, 3] = 2 * mu;
        R[4, 4] = -(3 * gamma + mu);
        R[4, 5] = 4 * gamma;

        R[5, 1] = 0;
        R[5, 2] = 0;
        R[5, 3] = 0;
        R[5, 4] = mu;
        R[5, 5] = -4 * gamma;

        return R;
     }

    vector diploid_prob(real t, vector initial_state, real mu, real gamma) {
        matrix[3, 3] R = dip_rate_matrix(mu, gamma);
        matrix[3, 1] init_mat;
        init_mat[, 1] = initial_state;
        matrix[3, 1] F = scale_matrix_exp_multiply(t, R, init_mat);
        return to_vector(F);
    }


    vector tri_prob(real t, vector initial_state, real mu, real gamma) {
        matrix[4, 4] R = tri_rate_matrix(mu, gamma);  
        matrix[4, 1] initial_state_matrix;
        initial_state_matrix[, 1] = initial_state;
        matrix[4, 1] F = scale_matrix_exp_multiply(t, R, initial_state_matrix);
        return to_vector(F);
    }

    vector tet_prob(real t, vector initial_state, real mu, real gamma) {
        matrix[5, 5] R = tet_rate_matrix(mu, gamma); 
        matrix[5, 1] initial_state_matrix;
        initial_state_matrix[, 1] = initial_state;
        matrix[5, 1] F = scale_matrix_exp_multiply(t, R, initial_state_matrix);
        return to_vector(F);
    }

    vector cnLOH_event_prob(vector probabilities) {
        real m_prob = probabilities[1];
        real k_prob = probabilities[2];
        real w_prob = probabilities[3];

        m_prob += 0.5 * k_prob;
        w_prob += 0.5 * k_prob;
        k_prob = 0;

        vector[3] initial_cnLOH_probs;
        initial_cnLOH_probs[1] = m_prob;
        initial_cnLOH_probs[2] = k_prob;
        initial_cnLOH_probs[3] = w_prob;

        return initial_cnLOH_probs;
        }

    vector tri_event_prob(vector probabilities) {
        real m_prob = probabilities[1];
        real k_prob = probabilities[2];
        real w_prob = probabilities[3];

        real u_prob = 0.5 * k_prob;  // Up event
        real d_prob = 0.5 * k_prob;  // Down event

        vector[4] initial_trisomy_probs;
        initial_trisomy_probs[1] = m_prob;
        initial_trisomy_probs[2] = u_prob;
        initial_trisomy_probs[3] = d_prob;
        initial_trisomy_probs[4] = w_prob;

        return initial_trisomy_probs;
    }
    vector tet_event_prob(vector probabilities) {
        real m_prob = probabilities[1];  
        real d_prob = probabilities[2]; 
        real w_prob = probabilities[3];  

        real k_prob = 0;  
        real v_prob = 0; 

        vector[5] initial_tetraploidy_probs;
        initial_tetraploidy_probs[1] = m_prob;
        initial_tetraploidy_probs[2] = k_prob;
        initial_tetraploidy_probs[3] = d_prob;
        initial_tetraploidy_probs[4] = v_prob;
        initial_tetraploidy_probs[5] = w_prob;

        return initial_tetraploidy_probs;
    }

    vector ss_cnloh_evln(real t, real age, real mu, real gamma) {

        vector[3] initial_state = ss_init_prob(mu, gamma);
        vector[3] post_cnLOH_state_probs = cnLOH_event_prob(initial_state);
        vector[3] altered_state_probs = diploid_prob(age - t, post_cnLOH_state_probs, mu, gamma);

        return altered_state_probs;
    }
    vector ss_tri_evln(real t, real age, real mu, real gamma) {

        vector[3] initial_state = ss_init_prob(mu, gamma);
        vector[4] post_tri_state_probs = tri_event_prob(initial_state);
        vector[4] altered_state_probs = tri_prob(age - t, post_tri_state_probs, mu, gamma);

        return altered_state_probs;
    }

    vector ss_tet_evln(real t, real age, real mu, real gamma) {

        vector[3] initial_state = ss_init_prob(mu, gamma);
        vector[5] post_tet_state_probs = tet_event_prob(initial_state);
        vector[5] altered_state_probs = tet_prob(age - t, post_tet_state_probs, mu, gamma);

        return altered_state_probs;
    }

    vector dip_cnloh_evln(real t, real age, real mu, real gamma, vector initial_state) {
       
        vector[3] state_probs = diploid_prob(t, initial_state, mu, gamma);
        vector[3] post_cnloh_state_probs = cnLOH_event_prob(state_probs);
        vector[3] altered_state_probs = diploid_prob(age - t, post_cnloh_state_probs, mu, gamma);

        return altered_state_probs;
    }

    vector dip_tri_evln(real t, real age, real mu, real gamma, vector initial_state) {

        vector[3] state_probs = diploid_prob((t), initial_state, mu, gamma);
        vector[4] post_tri_state_probs = tri_event_prob(state_probs);
        vector[4] altered_state_probs = tri_prob(age - t, post_tri_state_probs, mu, gamma);

        return altered_state_probs;
    }

    vector dip_tet_evln(real t, real age, real mu, real gamma, vector initial_state) {

        vector[3] state_probs = diploid_prob((t), initial_state, mu, gamma);
        vector[5] post_tet_state_probs = tet_event_prob(state_probs);
        vector[5] altered_state_probs = tet_prob(age - t, post_tet_state_probs, mu, gamma);

        return altered_state_probs;
    }
}

data{
    int<lower=1> K;                        //number of mixture components
    int<lower=0> J;                        //number of patients
    array[J] int<lower=0> n;                       //number of sites for each patient
    array[sum(n)] real<lower=0,upper=1> y;     //observed beta values
    array[J] int<lower=0> age;                 //patient age
    int<lower=1,upper=6> type;

}

transformed data {
    ordered[K+1] position;
    array[sum(n)] real rand_vals; 
    array[sum(n)] vector[3] initial_state;   // Initial state per site
    array[J+1] int start_idx;
   
    if (type==1 || type==4){
        position[1] = 0.0;
        position[2] = 0.5;
        position[3] = 1.0;
    } else if (type==2 || type==5){
        position[1] = 0.0;
        position[2] = 1.0/3.0;
        position[3] = 2.0/3.0;
        position[4] = 1.0;
    } else if (type==3 || type==6){
        position[1] = 0.0;
        position[2] = 1.0/4.0;
        position[3] = 2.0/4.0;
        position[4] = 3.0/4.0;
        position[5] = 1.0;
    }


    start_idx[1] = 1; 
    for (j in 1:J) {
        start_idx[j+1] = start_idx[j] + n[j];
    }

    for (i in 1:sum(n)) {
        rand_vals[i] = uniform_rng(0, 1);
    }

    for (i in 1:sum(n)) {
        if (type == 4 || type == 5 || type == 6) {
            if (rand_vals[i] > 0.5) {
                initial_state[i] = [1, 0, 0]'; 
            } else {
                initial_state[i] = [0, 0, 1]';  
            }
        }
    }

}


parameters{
    array[J] real<lower=0> kappa;                //standard deviation of peak
    real<lower=0> mu;
    array[J] real<lower=0,upper=1> eta;          //Beta dist parameter
    array[J] real<lower=0,upper=1> delta;        //Beta dist parameter
    real<lower=0> gamma_raw;
    array[J] real<lower=0, upper=1> t_raw;  

    

}

transformed parameters {

    array[J, K+1] real<lower=0> a;            // Beta distribution shape parameter a
    array[J, K+1] real<lower=0> b;           // Beta distribution shape parameter b
    array[J] vector[K+1] cache_theta;       // Site-specific state probabilities
    real gamma;
    array[J] real t;

    gamma = gamma_raw * mu;


    for (j in 1:J) {
        vector[K+1] pos_obs;
        t[j] = t_raw[j] * age[j]; 
    

        for (i in 1:K+1) {
            pos_obs[i] = (eta[j] - delta[j]) * position[i] + delta[j];
            a[j, i] = kappa[j] * pos_obs[i]; 
            b[j, i] = kappa[j] * (1 - pos_obs[i]);
        }


        if (type == 1) {
            cache_theta[j] = ss_cnloh_evln(t[j], age[j], mu, gamma);
        } else if (type == 2) {
            cache_theta[j] = ss_tri_evln(t[j], age[j], mu, gamma);
        } else if (type == 3) {
            cache_theta[j] = ss_tet_evln(t[j], age[j], mu, gamma);
        } else if (type == 4) {
            cache_theta[j] = dip_cnloh_evln(t[j], age[j], mu, gamma, [0.5,0,0.5]');
        } else if (type == 5) {
            cache_theta[j] = dip_tri_evln(t[j], age[j], mu, gamma, [0.5,0,0.5]');
        } else if (type == 6) {
            cache_theta[j] = dip_tet_evln(t[j], age[j], mu, gamma, [0.5,0,0.5]');
        }         
    }
}
    


model {
    mu ~ normal(0,0.05);
    gamma_raw ~ normal(1,0.5);

    for (j in 1:J) {
        kappa[j] ~ lognormal(3.6, 0.5);       
        delta[j] ~ beta(5, 95);
        eta[j] ~ beta(95, 5);

        for (i in start_idx[j]:(start_idx[j+1] - 1)) {
            vector[K+1] log_likelihood = log(cache_theta[j]);

            for (k in 1:K+1) {
                log_likelihood[k] += beta_lpdf(y[i] | a[j, k], b[j, k]);
            }

            target += log_sum_exp(log_likelihood);
        }
    }
}

generated quantities {
    vector[sum(n)] y_rep;         // Posterior predictive samples (flat vector)
    vector[sum(n)] log_lik;       // Log-likelihood values (flat vector)
    array[sum(n), K+1] real log_lik_sep;  // Log-likelihood values for each state per sample

    for (j in 1:J) {
        for (i in start_idx[j]:(start_idx[j+1] - 1)) {
            // Generate replicated data for each patient
            int k_sim = categorical_rng(softmax(cache_theta[j]));
            y_rep[i] = beta_rng(a[j, k_sim], b[j, k_sim]);

            // Calculate log-likelihood
            vector[K+1] log_likelihood = log(cache_theta[j]);
            for (k in 1:K+1) {
                log_likelihood[k] += beta_lpdf(y[i] | a[j, k], b[j, k]);
            }
            log_lik[i] = log_sum_exp(log_likelihood);
            
            for (k in 1:K+1) {
                log_lik_sep[i, k] = log_likelihood[k];
            }
        }
    }
}
        
