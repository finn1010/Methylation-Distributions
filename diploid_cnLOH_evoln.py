import numpy as np
import matplotlib.pyplot as plt
from state_evolve.diploid_evolution import run_simulation_diploid, diploid_beta_vals, diploid_prob_matrix
# from state_evolve.tetraploidy_evolution import run_simulation_tetraploidy, tetraploidy_beta_vals, tetraploidy_prob_matrix
from state_evolve.cnLOH_event import cnLOH_event, cnLOH_event_prob
from plot import hist_plot

initial_state = np.array([1, 0, 0])  
mu = 0.02                          
gamma = 0.02                 
time_points = np.linspace(0, 200, 100) 
event_time = 140
evoln_time = 350


def diploid_to_cnLOH_hist(mu, gamma, num_cells, pre_time, post_time):
    diploid_states = run_simulation_diploid(mu, gamma, num_cells, pre_time, initial_state=None)
    beta_vals = []
    beta_vals.append(diploid_beta_vals(diploid_states))
    cnLOH_initial_state = cnLOH_event(diploid_states[-1])
    cnLOH_states = run_simulation_diploid(mu, gamma, num_cells, post_time, cnLOH_initial_state)
    beta_vals.append(diploid_beta_vals(cnLOH_states))

    hist_plot(beta_vals)

diploid_to_cnLOH_hist(mu, gamma, 100,100, 100)


def diploid_to_cnLOH_prob_dist(initial_state, mu, gamma, event_time, evoln_time):

    diploid_evoln_time = np.linspace(0,event_time)
    diploid_probs = diploid_prob_matrix(initial_state, mu, gamma, diploid_evoln_time)
    initial_cnLOH_probs = cnLOH_event_prob(diploid_probs)

    cnLOH_evoln_time = np.linspace(0,evoln_time-event_time)
    cnLOH_probs = diploid_prob_matrix(initial_cnLOH_probs, mu, gamma, cnLOH_evoln_time)
    initial_cnLOH_probs = np.array([cnLOH_event_prob(diploid_probs)])
 
    for i in range(diploid_probs.shape[1]):
        plt.plot(diploid_evoln_time, diploid_probs[:, i])
 
    for i in range(cnLOH_probs.shape[1]):
        plt.plot(cnLOH_evoln_time + event_time, cnLOH_probs[:, i])

    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.title('Diploid to cnLOH Evolution')
    plt.legend()
    plt.grid()
    plt.show()
    

diploid_to_cnLOH_prob_dist(initial_state, mu, gamma, event_time, evoln_time)
