import numpy as np
import matplotlib.pyplot as plt
from state_evolve.diploid_evolution import run_simulation_diploid, diploid_beta_vals, diploid_prob_matrix, ss_initialisation, state_initialisation
from state_evolve.tetraploidy_evolution import run_simulation_tetraploidy, tetraploidy_beta_vals, tetraploidy_prob_matrix
from state_evolve.tetraploidy_event import tetraploidy_event, tetraploidy_event_prob
from plot import hist_plot
from colours import pallet_dip, pallet_tet
               


def diploid_to_tetraploidy_hist(mu, gamma, init_fn, num_sites, event_time, patient_age,fig_name):
    final_diploid_states = run_simulation_diploid(mu, gamma,init_fn, num_sites, start_evoln=0, end_evoln=event_time, initial_state=None)
    beta_vals_before = []
    beta_vals_after = []
    beta_vals_before.append(diploid_beta_vals(final_diploid_states))
    tetraploidy_state_list = []
    for state in final_diploid_states:
        tetraploidy_initial_state = tetraploidy_event(state)
        tetraploidy_states = run_simulation_tetraploidy(mu, gamma, tetraploidy_initial_state, start_evoln=0, end_evoln=patient_age-event_time)
        tetraploidy_state_list.extend(tetraploidy_states)
    beta_vals_after.append(tetraploidy_beta_vals(tetraploidy_state_list))
 
    hist_plot(beta_vals_before, beta_vals_after,'Tetraploidy', event_time, patient_age-event_time,fig_name)

mu = 0.02                          
gamma = 0.02  
diploid_to_tetraploidy_hist(mu, gamma, ss_initialisation, 1000, 10, 60,'/Users/finnkane/Desktop/ICR/plots/Tetraploidy/Hist/ss_tet_hist_tau=10')
diploid_to_tetraploidy_hist(mu, gamma, ss_initialisation, 1000, 50, 60,'/Users/finnkane/Desktop/ICR/plots/Tetraploidy/Hist/ss_tet_hist_tau=50')

diploid_to_tetraploidy_hist(mu, gamma, state_initialisation, 1000, 10, 60,'/Users/finnkane/Desktop/ICR/plots/Tetraploidy/Hist/tet_hist_tau=10')
diploid_to_tetraploidy_hist(mu, gamma, state_initialisation, 1000, 50, 60,'/Users/finnkane/Desktop/ICR/plots/Tetraploidy/Hist/tet_hist_tau=50')


def diploid_to_tetraploidy_prob_dist(initial_state, mu, gamma, event_time, evoln_time, fig_name):

    diploid_evoln_time = np.linspace(0,event_time)
    diploid_probs = diploid_prob_matrix(initial_state, mu, gamma, diploid_evoln_time)
    initial_tetraploidy_probs = tetraploidy_event_prob(diploid_probs)

    tetraploidy_evoln_time = np.linspace(0,evoln_time-event_time)
    tetraploidy_probs = tetraploidy_prob_matrix(initial_tetraploidy_probs, mu, gamma, tetraploidy_evoln_time)
    initial_tetraploidy_probs = np.array([tetraploidy_event_prob(diploid_probs)])
    
    methylated_dip = [2,1,0]
    for i in range(diploid_probs.shape[1]):
        plt.plot(diploid_evoln_time, diploid_probs[:, i],label=f'Diploid: {methylated_dip[i]} Methylated Alleles', color=pallet_dip[i])

    methylated_tet = [4,3,2,1,0]
    for i in range(tetraploidy_probs.shape[1]):
        plt.plot(tetraploidy_evoln_time + event_time, tetraploidy_probs[:, i], label=f'Tetraploidy: {methylated_tet[i]} Methylated Alleles', color=pallet_tet[i])

    plt.axvline(x=event_time, color='gray', linestyle='--', label=f'τ={event_time}')

    plt.xlabel('Time (years)')
    plt.ylabel('Probability')
    plt.title('Diploid to Tetraploidy Evolution')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid()
    plt.savefig(f'{fig_name}.pdf', format='pdf', dpi=300)
    plt.show()
    
initial_state = ss_initialisation(mu,gamma)
initial_state = [0.5,0,0.5]
num_sites = 10000
event_time = 10
evoln_time = 60
diploid_to_tetraploidy_prob_dist(initial_state, mu, gamma, event_time, evoln_time, '/Users/finnkane/Desktop/ICR/plots/Tetraploidy/Prob/tet_prob_tau=10')
event_time = 50
evoln_time = 60
diploid_to_tetraploidy_prob_dist(initial_state, mu, gamma, event_time, evoln_time, '/Users/finnkane/Desktop/ICR/plots/Tetraploidy/Prob/tri_prob_tau=50')