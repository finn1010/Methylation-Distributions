
data {
    int<lower=0> N;         // Number of data points
    vector[N] x;            // Dosage levels
    vector[N] y;            // Blood pressure reductions
}
parameters {
    real alpha;             // Baseline reduction (intercept)
    real beta;              // Effect per mg of dosage (slope)
    real<lower=0> sigma;    // Variability in response
}
model {
    // Priors
    alpha ~ normal(2, 1);         // Prior: small reduction at zero dosage
    beta ~ normal(1, 0.5);        // Prior: expected effect of drug per mg
    sigma ~ cauchy(0, 2);         // Prior: variability in response

    // Likelihood
    y ~ normal(alpha + beta * x, sigma);
}
