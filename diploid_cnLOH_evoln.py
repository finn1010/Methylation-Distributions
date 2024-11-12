import numpy as np
import matplotlib.pyplot as plt
from state_evolve.diploid_evolution import run_simulation_diploid, diploid_beta_vals, diploid_prob_matrix
from state_evolve.cnLOH_event import cnLOH_event, cnLOH_event_prob
from plot import hist_plot

initial_state = np.array([1, 0, 0])  
mu = 0.02                          
gamma = 0.02                 
time_points = np.linspace(0, 200, 100) 



def diploid_to_cnLOH_hist(mu, gamma, num_cells, pre_time, post_time):
    final_diploid_states = run_simulation_diploid(mu, gamma, num_cells, pre_time, initial_state=None)
    beta_vals = []
    beta_vals.append(diploid_beta_vals(final_diploid_states))
    cnLOH_state_list = []
    for state in final_diploid_states:
        cnLOH_initial_state = cnLOH_event(state)
        cnLOH_states = run_simulation_diploid(mu, gamma, num_cells, post_time, cnLOH_initial_state)
        cnLOH_state_list.extend(cnLOH_states)
    beta_vals.append(diploid_beta_vals(cnLOH_state_list))

    hist_plot(beta_vals)

diploid_to_cnLOH_hist(mu, gamma, 100,10,50)
diploid_to_cnLOH_hist(mu, gamma, 100,50,10)



def diploid_to_cnLOH_prob_dist(initial_state, mu, gamma, event_time, evoln_time):

    diploid_evoln_time = np.linspace(0,event_time)
    diploid_probs = diploid_prob_matrix(initial_state, mu, gamma, diploid_evoln_time)
    initial_cnLOH_probs = cnLOH_event_prob(diploid_probs)

    cnLOH_evoln_time = np.linspace(0,evoln_time-event_time)
    cnLOH_probs = diploid_prob_matrix(initial_cnLOH_probs, mu, gamma, cnLOH_evoln_time)
    initial_cnLOH_probs = np.array([cnLOH_event_prob(diploid_probs)])
 
    for i in range(diploid_probs.shape[1]):
        plt.plot(diploid_evoln_time, diploid_probs[:, i], label=f'Diploid State {i+1}')
 
    for i in range(cnLOH_probs.shape[1]):
        plt.plot(cnLOH_evoln_time + event_time, cnLOH_probs[:, i], label=f'cnLOH State {i+1}')

    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.title('Diploid to cnLOH Evolution')
    plt.legend()
    plt.grid()
    plt.show()
    
event_time = 10
evoln_time = 60
diploid_to_cnLOH_prob_dist(initial_state, mu, gamma, event_time, evoln_time)
event_time = 50
evoln_time = 60
diploid_to_cnLOH_prob_dist(initial_state, mu, gamma, event_time, evoln_time)
