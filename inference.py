import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
from state_evolve.diploid_evolution import ss_initialisation, state_initialisation
from cnLOH_inf import diploid_to_cnLOH
from tri_inf import diploid_to_trisomy
from tet_inf import diploid_to_tetraploidy
import numpy as np
# Model parameters
mu = 0.02
gamma = 0.02    
num_sites = 1000
event_time = 50
patient_age = 60
type = 4

# Simulate data based on type
if type == 1:
    noisy_beta_before, noisy_beta_after = diploid_to_cnLOH(mu, gamma, ss_initialisation, num_sites, event_time, patient_age)
    K = 2   
elif type == 2:
    noisy_beta_before, noisy_beta_after = diploid_to_trisomy(mu, gamma, ss_initialisation, num_sites, event_time, patient_age)
    K = 3
elif type == 3:
    noisy_beta_before, noisy_beta_after = diploid_to_tetraploidy(mu, gamma, ss_initialisation, num_sites, event_time, patient_age)
    K = 4
elif type == 4:
    noisy_beta_before, noisy_beta_after = diploid_to_cnLOH(mu, gamma, state_initialisation, num_sites, event_time, patient_age)
    K = 2
elif type == 5:
    noisy_beta_before, noisy_beta_after = diploid_to_trisomy(mu, gamma, state_initialisation, num_sites, event_time, patient_age)
    K = 3
elif type == 6:
    noisy_beta_before, noisy_beta_after = diploid_to_tetraploidy(mu, gamma, state_initialisation, num_sites, event_time, patient_age)
    K = 4

# Compile and sample from the Stan model
model = cmdstanpy.CmdStanModel(stan_file='multi_inf.stan')
data = {
    'K': K, 
    'N': num_sites, 
    'y': noisy_beta_after,  
    'age': patient_age,
    'type': type
}

fit = model.sample(data=data, show_console=True)
print(fit.summary())

# Extract inference diagnostics
az_fit = az.from_cmdstanpy(fit, posterior_predictive=['y_rep'], observed_data={'y': noisy_beta_after})

# Plot diagnostics
az.plot_posterior(az_fit, var_names=["mu", "gamma", "t"])
plt.title('Posterior Estimates')
plt.show()

az.plot_forest(az_fit, var_names=["t"], combined=False)
plt.title('Forest Plot of Posterior Estimates for t')
plt.show()

az.plot_forest(az_fit, var_names=["mu", "gamma"], combined=False)
plt.title('Forest Plot of Posterior Estimates for mu and gamma')
plt.show()

az.plot_pair(az_fit, var_names=["mu", "gamma", "t"], kind="kde")
plt.title('Pair Plot of Posterior Estimates')
plt.show()

az.plot_trace(az_fit, var_names=["mu", "gamma", "t"], combined=True)
plt.title('Trace Plot for mu, gamma, and t')
plt.show()

az.plot_rank(az_fit, var_names=["mu", "gamma", "t"], kind="vlines")
plt.title('Rank Plot for mu, gamma, and t')
plt.show()

az.plot_ppc(az_fit, kind="kde", observed=True, data_pairs={"y": "y_rep"})
plt.title('Posterior Predictive Check')
plt.show()

posterior_predictive = az_fit.posterior_predictive["y_rep"].values
observed = az_fit.observed_data["y"].values
lower_bound = np.percentile(posterior_predictive, 2.5, axis=0)
upper_bound = np.percentile(posterior_predictive, 97.5, axis=0)
coverage = np.mean((observed >= lower_bound) & (observed <= upper_bound)) * 100
print(f"Credible Interval Coverage: {coverage:.2f}%")

# 2. Variance Comparison
variance_observed = np.var(observed)
variance_posterior = np.var(posterior_predictive.mean(axis=0))
print(f"Variance of Observed Data: {variance_observed:.4f}")
print(f"Variance of Posterior Predictive: {variance_posterior:.4f}")
print(f"Variance Ratio (Posterior/Observed): {variance_posterior / variance_observed:.2f}")

# 3. Histogram Overlap
plt.hist(observed, bins=30, alpha=0.5, label="Observed", density=True)
plt.hist(posterior_predictive.flatten(), bins=30, alpha=0.5, label="Posterior Predictive", density=True)
plt.legend()
plt.title('Histogram of Observed vs Posterior Predictive Data')
plt.xlabel('Beta Values')
plt.ylabel('Density')
plt.show()