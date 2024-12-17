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
type = 6


if type == 1:
    noisy_beta_before, noisy_beta_after = diploid_to_cnLOH(mu, gamma, ss_initialisation, num_sites, event_time, patient_age)
    K = 2
    
    prefix = f'/Users/finnkane/Desktop/ICR/inf_plots/ss_cnloh/t={event_time}/'
elif type == 2:
    noisy_beta_before, noisy_beta_after = diploid_to_trisomy(mu, gamma, ss_initialisation, num_sites, event_time, patient_age)
    K = 3
    prefix = f'/Users/finnkane/Desktop/ICR/inf_plots/ss_tri/t={event_time}/'
elif type == 3:
    noisy_beta_before, noisy_beta_after = diploid_to_tetraploidy(mu, gamma, ss_initialisation, num_sites, event_time, patient_age)
    K = 4
    prefix = f'/Users/finnkane/Desktop/ICR/inf_plots/ss_tet/t={event_time}/'
elif type == 4:
    noisy_beta_before, noisy_beta_after = diploid_to_cnLOH(mu, gamma, state_initialisation, num_sites, event_time, patient_age)
    K = 2
    prefix = f'/Users/finnkane/Desktop/ICR/inf_plots/dip_cnloh/t={event_time}/'
elif type == 5:
    noisy_beta_before, noisy_beta_after = diploid_to_trisomy(mu, gamma, state_initialisation, num_sites, event_time, patient_age)
    K = 3
    prefix = f'/Users/finnkane/Desktop/ICR/inf_plots/dip_tri/t={event_time}/'
elif type == 6:
    noisy_beta_before, noisy_beta_after = diploid_to_tetraploidy(mu, gamma, state_initialisation, num_sites, event_time, patient_age)
    K = 4
    prefix = f'/Users/finnkane/Desktop/ICR/inf_plots/dip_tet/t={event_time}/'

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
plt.savefig(f'{prefix}posterior.pdf', format='pdf', dpi=300)
plt.show()

az.plot_forest(az_fit, var_names=["t"], combined=False)
plt.title('Forest Plot of Posterior Estimates for t')
plt.savefig(f'{prefix}forest.pdf', format='pdf', dpi=300)
plt.show()

az.plot_forest(az_fit, var_names=["mu", "gamma"], combined=False)
plt.title('Forest Plot of Posterior Estimates for mu and gamma')
plt.savefig(f'{prefix}forest.pdf', format='pdf', dpi=300)
plt.show()

az.plot_pair(az_fit, var_names=["mu", "gamma", "t"])
plt.title('Pair Plot of Posterior Estimates')
plt.savefig(f'{prefix}pair_plot.pdf', format='pdf', dpi=300)
plt.show()

az.plot_trace(az_fit, var_names=["mu", "gamma", "t"], combined=True)
# plt.title('Trace Plot for mu, gamma, and t')
plt.savefig(f'{prefix}trace_plot.pdf', format='pdf', dpi=300)
plt.show()

az.plot_rank(az_fit, var_names=["mu", "gamma", "t"])
# plt.title('Rank Plot for mu, gamma, and t')
plt.savefig(f'{prefix}rank_plot.pdf', format='pdf', dpi=300)
plt.show()

az.plot_ppc(az_fit, kind="kde", observed=True, data_pairs={"y": "y_rep"})
plt.title('Posterior Predictive Check')
plt.savefig(f'{prefix}PPC.pdf', format='pdf', dpi=300)
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

variance_observed = np.var(observed)
variance_posterior = np.var(posterior_predictive.mean(axis=0))
variance_ratio = variance_posterior / variance_observed

with open(f"{prefix}analysis_results.txt", "w") as file:
    file.write(f"Credible Interval Coverage: {coverage:.2f}%\n")
    file.write(f"Variance of Observed Data: {variance_observed:.4f}\n")
    file.write(f"Variance of Posterior Predictive: {variance_posterior:.4f}\n")
    file.write(f"Variance Ratio (Posterior/Observed): {variance_ratio:.2f}\n")

# 3. Histogram Overlap
plt.hist(observed, bins=30, alpha=0.5, label="Observed", density=True)
plt.hist(posterior_predictive.flatten(), bins=30, alpha=0.5, label="Posterior Predictive", density=True)
plt.legend()
plt.title('Histogram of Observed vs Posterior Predictive Data')
plt.xlabel('Beta Values')
plt.ylabel('Density')
plt.savefig(f'{prefix}hist_plot.pdf', format='pdf', dpi=300)
plt.show()