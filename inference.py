import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
from state_evolve.diploid_evolution import ss_initialisation, ss_init_prob
from cnLOH_inf import diploid_to_cnLOH
from state_evolve.diploid_evolution import ss_initialisation, ss_init_prob

mu = 0.02
gamma = 0.02
init_state = ss_initialisation(mu,gamma)
num_sites = 1000
event_time = 10
patient_age = 60
noisy_beta_before, noisy_beta_after = diploid_to_cnLOH(mu, gamma, ss_initialisation, num_sites, event_time, patient_age)


model = cmdstanpy.CmdStanModel(stan_file='inference.stan')
data = {
    'K': 2, 
    'N': num_sites, 
    'y': noisy_beta_after,  
    'age': patient_age 
}
fit = model.sample(data=data, iter_warmup=50, iter_sampling=50, show_console=True)
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