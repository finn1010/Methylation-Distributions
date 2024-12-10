import cmdstanpy
import numpy as np
import matplotlib.pyplot as plt

# Define the Stan model as a string
stan_model_code = """
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
"""
def x(l):
    for i in range(l,1):
        print(i)
# Save model to a .stan file
with open("linear_model.stan", "w") as file:
    file.write(stan_model_code)

# Compile the Stan model
model = cmdstanpy.CmdStanModel(stan_file="linear_model.stan")

# Generate synthetic data to simulate a clinical trial
np.random.seed(42)
N = 100                       # Number of patients
x = np.random.uniform(0, 10, N)  # Dosage levels from 0 to 10 mg
true_alpha = 2                # True baseline reduction
true_beta = 1                 # True effect per mg
true_sigma = 2                # True variability in response
y = true_alpha + true_beta * x + np.random.normal(0, true_sigma, N)

# Package the data for Stan
data = {
    'N': N,
    'x': x,
    'y': y
}

# Sample from the posterior
fit = model.sample(data=data, chains=4, iter_sampling=2000, iter_warmup=500)

# Print a summary of the posterior
print(fit.summary())

# Extract samples for further analysis
alpha_samples = fit.stan_variable('alpha')
beta_samples = fit.stan_variable('beta')
sigma_samples = fit.stan_variable('sigma')

# Posterior means of parameters
alpha_mean = np.mean(alpha_samples)
beta_mean = np.mean(beta_samples)
sigma_mean = np.mean(sigma_samples)
print(f"Posterior means:\n alpha: {alpha_mean}, beta: {beta_mean}, sigma: {sigma_mean}")

# Plot the data and the fitted regression line using posterior means
plt.scatter(x, y, label="Data")
plt.plot(x, alpha_mean + beta_mean * x, color='red', label="Fitted line")
plt.xlabel("Dosage (mg)")
plt.ylabel("Blood Pressure Reduction (mmHg)")
plt.legend()
plt.show()
