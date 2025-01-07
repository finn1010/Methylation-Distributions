import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
from state_evolve.diploid_evolution import ss_initialisation, state_initialisation
from cnLOH_inf import diploid_to_cnLOH
from tri_inf import diploid_to_trisomy
from tet_inf import diploid_to_tetraploidy
import numpy as np
import pandas as pd
from plot import hist_plot
from trial_dip import cnloh_sim

# J=4
# num_sites = np.random.uniform(100, 200, size=J).astype(int)
# patient_ages = np.random.uniform(60, 80, size=J).astype(int)
# event_times=[]
# for patient in patient_ages:
#     event_times.append(np.random.uniform(10, patient-10))


# Model parameters
mu = 0.001
gamma = 0.001
type = 4
event_times = [20,35,10,15,60]
J=5
num_sites=[600,600,600,600,600]
patient_ages=[30,40,55,80,70]



vals = []
if type == 1:
    for i in range(len(patient_ages)):
        noisy_beta_before, noisy_beta_after = diploid_to_cnLOH(mu, gamma, ss_initialisation, num_sites[i], event_times[i], patient_ages[i])
        vals.append(noisy_beta_after)
        K=2
    # prefix = f'/Users/finnkane/Desktop/ICR/inf_plots/ss_cnloh/t={event_time}/'
elif type == 2:
    for i in range(len(patient_ages)):
        noisy_beta_before, noisy_beta_after = diploid_to_trisomy(mu, gamma, ss_initialisation, num_sites[i], event_times[i], patient_ages[i])
        K = 3
    # prefix = f'/Users/finnkane/Desktop/ICR/inf_plots/ss_tri/t={event_time}/'
elif type == 3:
    for i in range(len(patient_ages)):
        noisy_beta_before, noisy_beta_after = diploid_to_tetraploidy(mu, gamma, ss_initialisation, num_sites[i], event_times[i], patient_ages[i])
        K = 4
    # prefix = f'/Users/finnkane/Desktop/ICR/inf_plots/ss_tet/t={event_time}/'
elif type == 4:
    for i in range(len(patient_ages)):
        noisy_beta_after1 = cnloh_sim([0.5,0,0.5], mu, gamma, patient_ages[i], event_times[i], num_sites[i])
        #noisy_beta_before, noisy_beta_after = diploid_to_cnLOH(mu, gamma, state_initialisation, num_sites[i], event_times[i], patient_ages[i])
        vals.append(noisy_beta_after1)
        # hist_plot(noisy_beta_before, noisy_beta_after,'Diploid', event_times[i], patient_ages[i]-event_times[i], 'e.png')
        # hist_plot(noisy_beta_after1, noisy_beta_after1,'Diploid', event_times[i], patient_ages[i]-event_times[i], 'e.png')

        K = 2
    # prefix = f'/Users/finnkane/Desktop/ICR/inf_plots/dip_cnloh/t={event_time}/'
elif type == 5:
    for i in range(len(patient_ages)):
        noisy_beta_before, noisy_beta_after = diploid_to_trisomy(mu, gamma, state_initialisation, num_sites[i], event_times[i], patient_ages[i])
        vals.append(noisy_beta_after)
        K = 3
    # prefix = f'/Users/finnkane/Desktop/ICR/inf_plots/dip_tri/t={event_time}/'
elif type == 6:
    for i in range(len(patient_ages)):
        noisy_beta_before, noisy_beta_after = diploid_to_tetraploidy(mu, gamma, state_initialisation, num_sites[i], event_times[i], patient_ages[i])
        K = 4
    # prefix = f'/Users/finnkane/Desktop/ICR/inf_plots/dip_tet/t={event_time}/'
        vals.append(noisy_beta_after)

vals = np.concatenate(vals)

model = cmdstanpy.CmdStanModel(stan_file='multi_inf.stan')
data = {
    'K': K, 
    'J': J,
    'n': num_sites, 
    'y': vals,  
    'age': patient_ages,
    'type': type,
}

init_values = [{'t': np.random.uniform(1, 70, size=J).tolist()} for _ in range(4)]

fit = model.sample(
    data=data,
    iter_sampling=1000,  
    iter_warmup=1000,  
    adapt_delta=0.8,
    max_treedepth=12,
    inits=init_values,
    show_console=False
)
summary_df = fit.summary()

filtered_summary = summary_df[summary_df.index.str.contains(r"^(t(\[\d+\]|_raw\[\d+\])?|mu|gamma)$", regex=True)]
print(filtered_summary)

print(event_times)

divergences = fit.diagnose()
print(divergences)

#
az_fit = az.from_cmdstanpy(
    fit,
    posterior_predictive="y_rep",  
    observed_data={"y": vals}      
)

az.plot_pair(
    az_fit,
    var_names=["mu", "gamma_raw", "t"], 
    divergences=True,
)
az.plot_posterior(az_fit, var_names=["mu", "gamma", "t"])
plt.title('Posterior Estimates')
# plt.savefig(f'{prefix}posterior.pdf', format='pdf', dpi=300)
plt.show()

az.plot_trace(
    az_fit, 
    var_names=["mu", "gamma_raw", "t"], 
    divergences='top', 
    combined=True)
plt.show()

y_rep_flat = az_fit.posterior_predictive["y_rep"].stack(samples=("chain", "draw"))
print(y_rep_flat.shape)

print(az_fit.posterior_predictive.keys())
print(f"Observed data shape: {np.shape(vals)}")
print(f"Posterior predictive shape: {az_fit.posterior_predictive['y_rep'].shape}")

az_fit.posterior_predictive["y_rep"] = y_rep_flat

az.plot_ppc(
    data=az_fit,
    data_pairs={"y": "y_rep"}
)
plt.show()

# plt.savefig(f'cccc.pdf', format='pdf', dpi=300)
# az.plot_ppc(az_fit, var_names=["y_rep"])

# az.plot_autocorr(fit)

# print(event_times)
# # Check specific parameters with divergences

# az_fit = az.from_cmdstanpy(fit, posterior_predictive=['y_rep'], observed_data={'y': vals})

# # Plot diagnostics


# az.plot_forest(az_fit, var_names=["t"], combined=False)
# plt.title('Forest Plot of Posterior Estimates for t')
# # plt.savefig(f'{prefix}forest.pdf', format='pdf', dpi=300)
# plt.show()

# az.plot_forest(az_fit, var_names=["mu", "gamma"], combined=False)
# plt.title('Forest Plot of Posterior Estimates for mu and gamma')
# # plt.savefig(f'{prefix}forest.pdf', format='pdf', dpi=300)
# plt.show()

# az.plot_pair(az_fit, var_names=["mu", "gamma_raw", "t"])
# plt.title('Pair Plot of Posterior Estimates')
# # plt.savefig(f'{prefix}pair_plot.pdf', format='pdf', dpi=300)
# plt.show()

# az.plot_trace(az_fit, var_names=["mu", "gamma", "t"], combined=True)
# # plt.title('Trace Plot for mu, gamma, and t')
# plt.savefig(f'{prefix}trace_plot.pdf', format='pdf', dpi=300)
# plt.show()

# az.plot_rank(az_fit, var_names=["mu", "gamma", "t"])
# # plt.title('Rank Plot for mu, gamma, and t')
# plt.savefig(f'{prefix}rank_plot.pdf', format='pdf', dpi=300)
# plt.show()

# az.plot_ppc(az_fit, kind="kde", observed=True, data_pairs={"y": "y_rep"})
# plt.title('Posterior Predictive Check')
# # plt.savefig(f'{prefix}PPC.pdf', format='pdf', dpi=300)
# plt.show()

# posterior_predictive = az_fit.posterior_predictive["y_rep"].values
# observed = az_fit.observed_data["y"].values
# lower_bound = np.percentile(posterior_predictive, 2.5, axis=0)
# upper_bound = np.percentile(posterior_predictive, 97.5, axis=0)
# coverage = np.mean((observed >= lower_bound) & (observed <= upper_bound)) * 100
# print(f"Credible Interval Coverage: {coverage:.2f}%")

# # 2. Variance Comparison
# variance_observed = np.var(observed)
# variance_posterior = np.var(posterior_predictive.mean(axis=0))
# print(f"Variance of Observed Data: {variance_observed:.4f}")
# print(f"Variance of Posterior Predictive: {variance_posterior:.4f}")
# print(f"Variance Ratio (Posterior/Observed): {variance_posterior / variance_observed:.2f}")

# variance_observed = np.var(observed)
# variance_posterior = np.var(posterior_predictive.mean(axis=0))
# variance_ratio = variance_posterior / variance_observed

# # with open(f"{prefix}analysis_results.txt", "w") as file:
#     # file.write(f"Credible Interval Coverage: {coverage:.2f}%\n")
#     # file.write(f"Variance of Observed Data: {variance_observed:.4f}\n")
#     # file.write(f"Variance of Posterior Predictive: {variance_posterior:.4f}\n")
#     # file.write(f"Variance Ratio (Posterior/Observed): {variance_ratio:.2f}\n")

# # 3. Histogram Overlap
# plt.hist(observed, bins=30, alpha=0.5, label="Observed", density=True)
# plt.hist(posterior_predictive.flatten(), bins=30, alpha=0.5, label="Posterior Predictive", density=True)
# plt.legend()
# plt.title('Histogram of Observed vs Posterior Predictive Data')
# plt.xlabel('Beta Values')
# plt.ylabel('Density')
# # plt.savefig(f'{prefix}hist_plot.pdf', format='pdf', dpi=300)
# plt.show()