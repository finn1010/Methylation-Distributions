import numpy as np
import matplotlib.pyplot as plt

from state_evolve.diploid_evolution import run_simulation_diploid, diploid_beta_vals, diploid_prob_matrix
from state_evolve.trisomy_evolution import run_simulation_trisomy, trisomy_beta_vals, trisomy_prob_matrix
from state_evolve.trisomy_event import trisomy_event, trisomy_event_prob
from plot import hist_plot, plot_prob_dist
from colours import pallet_dip, pallet_tri
mu = 0.02                          
gamma = 0.02                 


def diploid_to_trisomy_hist(mu, gamma, num_sites, event_time, patient_age,fig_name):
    final_diploid_states = run_simulation_diploid(mu, gamma, num_sites, start_evoln=0, end_evoln=event_time, initial_state=None)
    beta_vals_before = []
    beta_vals_after = []
    beta_vals_before.append(diploid_beta_vals(final_diploid_states))
    trisomy_state_list = []
    for state in final_diploid_states:
        trisomy_initial_state = trisomy_event(state)
        trisomy_states = run_simulation_trisomy(mu, gamma, trisomy_initial_state, start_evoln=0, end_evoln=patient_age-event_time)
        trisomy_state_list.extend(trisomy_states)
    beta_vals_after.append(trisomy_beta_vals(trisomy_state_list))
 
    hist_plot(beta_vals_before, beta_vals_after,'Trisomy', event_time, patient_age-event_time,fig_name)

diploid_to_trisomy_hist(mu, gamma, 1000, 10, 60,'/Users/finnkane/Desktop/ICR/plots/Trisomy/Hist/tri_hist_tau=10')
diploid_to_trisomy_hist(mu, gamma, 1000, 50, 55,'/Users/finnkane/Desktop/ICR/plots/Trisomy/Hist/tri_hist_tau=50')

def diploid_to_trisomy_prob_dist(initial_state, mu, gamma, event_time, evoln_time, fig_name):

    diploid_evoln_time = np.linspace(0,event_time)
    diploid_probs = diploid_prob_matrix(initial_state, mu, gamma, diploid_evoln_time)
    initial_trisomy_probs = trisomy_event_prob(diploid_probs)

    trisomy_evoln_time = np.linspace(0,evoln_time-event_time)
    trisomy_probs = trisomy_prob_matrix(initial_trisomy_probs, mu, gamma, trisomy_evoln_time)
    initial_trisomy_probs = np.array([trisomy_event_prob(diploid_probs)])
    methylated_dip = [2, 1, 0]  
    methylated_tri = [3, 2, 1, 0]  
    for i in range(diploid_probs.shape[1]):
        plt.plot(diploid_evoln_time, diploid_probs[:, i], label=f'Diploid: {methylated_dip[i]} Methylated Alleles', color=pallet_dip[i])

    for i in range(trisomy_probs.shape[1]):
        plt.plot(trisomy_evoln_time + event_time, trisomy_probs[:, i], label=f'Trisomy: {methylated_tri[i]} Methylated Alleles', color=pallet_tri[i])

    plt.axvline(x=event_time, color='gray', linestyle='--', label=f'τ={event_time}')

    plt.xlabel('Time (years)')
    plt.ylabel('Probability')
    plt.title('Diploid to Trisomy Evolution')   
    plt.legend(loc='upper right', fontsize=9)
    plt.grid()
    plt.savefig(f'{fig_name}.pdf', format='pdf', dpi=300)
    plt.show()
    
event_time = 10
evoln_time = 60
diploid_to_trisomy_prob_dist(initial_state, mu, gamma, event_time, evoln_time,'/Users/finnkane/Desktop/ICR/plots/Trisomy/Prob/tri_prob_tau=10')
event_time = 50
evoln_time = 60
diploid_to_trisomy_prob_dist(initial_state, mu, gamma, event_time, evoln_time,'/Users/finnkane/Desktop/ICR/plots/Trisomy/Prob/tri_prob_tau=50')


