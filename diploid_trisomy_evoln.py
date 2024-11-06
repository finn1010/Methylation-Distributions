import numpy as np
import matplotlib.pyplot as plt

from state_evolve.diploid_evolution import run_simulation_diploid, diploid_beta_vals, diploid_prob_matrix
from state_evolve.trisomy_evolution import run_simulation_trisomy, trisomy_beta_vals, trisomy_prob_matrix
from state_evolve.trisomy_event import trisomy_event, trisomy_event_prob
from plot import hist_plot, plot_prob_dist
initial_state = np.array([1, 0, 0])  
mu = 0.02                          
gamma = 0.02                 
event_time = 50
evoln_time = 200

def diploid_to_trisomy_hist(initial_state, mu, gamma):
    diploid_states = run_simulation_diploid(initial_state, mu, gamma, num_iterations=50000)
    beta_vals = []
    beta_vals.append(diploid_beta_vals(diploid_states))
    trisomy_initial_state = trisomy_event(diploid_states[-1])
    trisomy_states = run_simulation_trisomy(trisomy_initial_state, mu, gamma, num_iterations=50000)
    beta_vals.append(trisomy_beta_vals(trisomy_states))
    hist_plot(beta_vals)

def diploid_to_trisomy_prob_dist(initial_state, mu, gamma, event_time, evoln_time):

    diploid_evoln_time = np.linspace(0,event_time)
    diploid_probs = diploid_prob_matrix(initial_state, mu, gamma, diploid_evoln_time)
    initial_trisomy_probs = trisomy_event_prob(diploid_probs)

    trisomy_evoln_time = np.linspace(0,evoln_time-event_time)
    trisomy_probs = trisomy_prob_matrix(initial_trisomy_probs, mu, gamma, trisomy_evoln_time)
    initial_trisomy_probs = np.array([trisomy_event_prob(diploid_probs)])
 
    for i in range(diploid_probs.shape[1]):
        plt.plot(diploid_evoln_time, diploid_probs[:, i])
 
    for i in range(trisomy_probs.shape[1]):
        plt.plot(trisomy_evoln_time + event_time, trisomy_probs[:, i])

    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.title('Diploid to Trisomy Evolution')
    plt.legend()
    plt.grid()
    plt.show()
   

diploid_to_trisomy_prob_dist(initial_state, mu, gamma, event_time, evoln_time)