import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
from state_evolve.diploid_evolution import ss_initialisation, ss_init_prob, state_initialisation
from cnLOH_inf import diploid_to_cnLOH
from state_evolve.diploid_evolution import ss_initialisation, ss_init_prob
from tri_inf import diploid_to_trisomy
from tet_inf import diploid_to_tetraploidy
mu = 0.02
gamma = 0.02
num_sites = 1000
event_time = 10
patient_age = 60
type = 4

#ss dip - cnLOH - diploid
if type == 1:
    noisy_beta_before, noisy_beta_after = diploid_to_cnLOH(mu, gamma, ss_initialisation, num_sites, event_time, patient_age)

#ss dip - tri - tri
if type == 2:
    noisy_beta_before, noisy_beta_after = diploid_to_trisomy(mu, gamma, ss_initialisation, num_sites, event_time, patient_age)

#ss dip - tet - tet
if type == 3:
    noisy_beta_before, noisy_beta_after = diploid_to_tetraploidy(mu, gamma, ss_initialisation, num_sites, event_time, patient_age)

#diploid - cnLOH - diploid
if type == 4:
    noisy_beta_before, noisy_beta_after = noisy_beta_before, noisy_beta_after = diploid_to_cnLOH(mu, gamma, state_initialisation, num_sites, event_time, patient_age)

#dip - tri - tri
if type == 5:
    noisy_beta_before, noisy_beta_after = diploid_to_trisomy(mu, gamma, state_initialisation, num_sites, event_time, patient_age)

#dip - tet -tet
if type == 6:
    noisy_beta_before, noisy_beta_after = diploid_to_tetraploidy(mu, gamma, state_initialisation, num_sites, event_time, patient_age)


model = cmdstanpy.CmdStanModel(stan_file='multi_inf.stan')
data = {
    'K': 2, 
    'N': num_sites, 
    'y': noisy_beta_after,  
    'age': patient_age,
    'type': type
}
fit = model.sample(data=data, show_console=True)
print(fit.summary())

az_fit = az.from_cmdstanpy(fit)
az.plot_posterior(az_fit, var_names=["mu", "gamma", "t"])
plt.show()

az.plot_forest(az_fit, var_names=["t"], combined=False)
plt.title('Forest plot of posterior estimates')
plt.show()

az.plot_forest(az_fit, var_names=["mu", "gamma"], combined=False)
plt.title('Forest plot of posterior estimates')
plt.show()

az.plot_pair(az_fit, var_names=["mu", "gamma","t"])
plt.title('Pair plot of posterior estimates')
plt.show()

az.plot_trace(az_fit, var_names=["mu", "gamma", "t"], combined=True)
plt.title('Trace Plot for mu, gamma, and t')
plt.show()

az.plot_rank(az_fit, var_names=["mu", "gamma", "t"])
plt.title('Rank Plot for mu, gamma, and t')
plt.show()