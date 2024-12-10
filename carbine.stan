functions {
    int count_greater_than(vector v, int value) {
        int count = 0;
        for (i in 1:num_elements(v)) {
            if (v[i] > value) {
                count += 1;
            }
        }
        return count;
    }
    // This function transforms a relative time constrained between
    // 0 and 1 to between 0 and (T_k - t_offset)
    vector transform_t(vector t_raw, vector ages, vector t_offset, array[] int s){
        vector[num_elements(t_raw)] t;
        int pos = 1;                            // counter for iterating through s[k]
        for (k in 1:num_elements(s)){
            t[pos:(pos+s[k]-1)]= (ages[k] - t_offset[k]) * t_raw[pos:(pos+s[k]-1)];
            pos += s[k];
        }
        return t;
    }
    // This function calculates the mean of the poisson coefficient
    // for the alpha peak on the log scale
    vector calculate_lambda1(vector rate_log, 
                            array[] int CNA, 
                            vector region_log, 
                            vector t, 
                            array[] int s){
        vector[num_elements(t)] lambda_log1;
        int pos = 1;   // counter for iterating through s[k]
        // Loop through each patient and each CNA event
        // and calculate the expected number of mutations
        // on the alpha peak, taking into account the 
        // type of CNA
        for (k in 1:num_elements(s)){
            for (i in 0:(s[k]-1)){
                // LOH
                if (CNA[pos+i] == 1){
                    lambda_log1[pos+i] = (rate_log[k] + 
                                region_log[pos+i] + 
                                log(t[pos+i]));
                }
                // Tetraploidy
                else if (CNA[pos+i] == 2){
                    lambda_log1[pos+i] = (rate_log[k] + 
                                region_log[pos+i] + 
                                log(2 * t[pos+i]));
                }
                // Trisomy
                else if (CNA[pos+i] == 3){
                    lambda_log1[pos+i] = (rate_log[k] + 
                                region_log[pos+i] + 
                                log(t[pos+i]));
                }
            }
            pos += s[k];
        }
        return lambda_log1;
    }
    // This function calculates the mean of the poisson coefficient
    // for the beta peak on the log scale
    vector calculate_lambda2(vector rate_log, 
                            array[] int CNA,
                            vector region_log, 
                            vector t, 
                            vector t_offset,
                            vector ages,
                            array[] int s){
        vector[num_elements(t)] lambda_log2;
        int pos = 1;   // counter for iterating through s[k]
        // Loop through each patient and each CNA event
        // and calculate the expected number of mutations
        // on the beta peak, taking into account the 
        // type of CNA
        for (k in 1:num_elements(s)){
            for (i in 0:(s[k]-1)){
                //LOH
                if (CNA[pos+i] == 1){
                    lambda_log2[pos+i] = (rate_log[k] + 
                                region_log[pos+i] + 
                                log(2 * (ages[k] - t[pos+i] - t_offset[k]))
                                );
                }
                // Tetraploidy
                else if (CNA[pos+i] == 2){
                    lambda_log2[pos+i] = (rate_log[k] + 
                                region_log[pos+i] + 
                                log(4 * (ages[k] - t[pos+i] - t_offset[k]))
                                );
                }
                // Trisomy
                else if (CNA[pos+i] == 3){
                    lambda_log2[pos+i] = (rate_log[k] + 
                                region_log[pos+i] + 
                                log(t[pos+i] + 3 * (ages[k] - t[pos+i] - t_offset[k]))
                                );
                }
            }
            pos += s[k];
        }
        return lambda_log2;
    }
    // This function calculates the mean of the poisson coefficient
    // for the beta peak on the log scale
    vector calculate_lambda_diploid(vector rate_log, 
                            vector diploid_region_log, 
                            array[] int ii_obs,
                            vector t_offset,
                            vector ages){
        vector[num_elements(ii_obs)] lambda_log_diploid;
        // Loop through each patient with diploid regions
        // and calculate the expected number of clonal 
        // diploid mutations observed on the genome
        for (k in 1:num_elements(ii_obs)){
                //LOH
            lambda_log_diploid[k] = (rate_log[ii_obs[k]] + 
                        diploid_region_log[k] + 
                        log(2 * (ages[ii_obs[k]] - t_offset[ii_obs[k]]))
                        );
        }
        return lambda_log_diploid;
    }
}
data {
    int<lower=1> N;                             // # of chromosomal events
    int<lower=0> K;                             // # of patients
    array[K] int s;                             // group sizes
    array[N] int<lower=0> alpha;                // # of SNVs prior to CNA
    array[N] int<lower=0> beta;                 // # of SNVs post CNA
    vector<lower=0>[K] mu_eff;                  // the "eff" mutation rate, derived from the 1/f tail
    vector<lower=0>[K] ages;                    // patient age
    vector<lower=0>[N] region_size;             // DNA length in Mbp
    array[N] int<lower=1,upper=3> CNA;          // An array indicating the type of CNA event 
                                                // (LOH:1, Tetraploidy:2, Trisomy:3)
    array[K] int<lower=0> diploid_counts;       // # of clonal diploid SNVs 
    vector<lower=0>[K] diploid_region_size;     // DNA length in Mbp of diploid region for observed patients
}
transformed data{
    vector[N] region_log;
    vector[K] mu_eff_log;
    int<lower=0> K_obs= count_greater_than(diploid_region_size, 0);  // Number of patients with observed diploid mutations
    array[K_obs] int<lower=1, upper=K> ii_obs;    // Indices of observed diploid data
    vector[K_obs] diploid_region_log;
    array[K_obs] int<lower=0> diploid_obs;        // # of clonal diploid SNVs 
    int ii = 1;
    real SminLog = log(10^2);                     // log(minimum effective population size)
    real SmaxLog = log(10^9);                     // log(maximum effective population size)
    for (k in 1:K){
        if (diploid_region_size[k] > 0){
            ii_obs[ii] = k;
            diploid_region_log[ii] = log(diploid_region_size[k]);
            diploid_obs[ii] = diploid_counts[k];
            ii += 1;
        }
    }
    // cache log of data values
    region_log = log(region_size);
    mu_eff_log = log(mu_eff);
}
parameters {
    real growth_log_mu;                         // the population mean log(effective growth rate)
    real<lower=0> growth_log_sigma;             // the population std log(effective growth rate)
    vector[K] growth_log_raw;                   // dummy growth_log 
    vector<lower=0, upper=1>[N] t_raw;          // relative time in a patient's life at which CNA occurred
    vector<lower=0,upper=1>[K] t_offset_raw;    // dummy time since MRCA shared between patients
}
transformed parameters{
    vector[N] t;                                // absolute time in a patient's life at which CNA occurred
    vector[K] rate_log;                         // log(patient specific mutation rate)
    vector[K] t_offset;                         // time since MRCA shared between patients 
    vector<lower=0>[K] pop_size_log;       // effective population size since the MRCA
    vector[N] lambda_log1;                      // log(mean number of mutations on the alpha peak)
    vector[N] lambda_log2;                      // log(mean number of mutations on the beta peak)
    vector[K] growth_log;                       // log(effective growth rate per year)
    vector[K] growth;                           // the fraction of cell divisions where the resulting 
                                                // lineages survive multiplied by the division rate
    vector[K_obs] lambda_log_diploid;           // log(mean number of mutations clonal diploid mutations)
    growth_log = growth_log_mu + growth_log_sigma * growth_log_raw;
    rate_log = mu_eff_log + growth_log;
    growth = exp(growth_log); 
    t_offset = t_offset_raw .* ages;
    pop_size_log = t_offset .* growth;
    // transform t_raw which is constrained to [0, 1] to t which
    // is constrained to [0, ages-t_offset]
    t = transform_t(t_raw, ages, t_offset, s);
    lambda_log1 = calculate_lambda1(rate_log, CNA, region_log, t, s);
    lambda_log2 = calculate_lambda2(rate_log, CNA, region_log, t, t_offset, ages, s);
    lambda_log_diploid = calculate_lambda_diploid(rate_log, diploid_region_log, 
                                    ii_obs, t_offset, ages);
}
model {
    // priors
    growth_log_mu ~ normal(1.5, 0.5);
    growth_log_sigma ~ normal(0, 0.5);
    growth_log_raw ~ normal(0, 1);
    t_raw ~ beta(2, 2);
    t_offset_raw ~ beta(1, 9);
    pop_size_log ~ normal((SminLog + SmaxLog) / 2,
                            (SmaxLog + SminLog) / 4);
    // model
    
    // the number of mutations on each peak is assumed to be poisson distributed
    // according to the mutation rate and the time at which the CNA occurred
    alpha ~ poisson_log(lambda_log1);
    beta ~ poisson_log(lambda_log2);
    diploid_obs ~ poisson_log(lambda_log_diploid);
}
generated quantities {
    vector[K] rate = exp(rate_log);             // patient specific mutation rate
}