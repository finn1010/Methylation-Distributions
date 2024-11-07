import numpy as np
import matplotlib.pyplot as plt
from state_evolve.diploid_evolution import run_simulation_diploid, diploid_beta_vals, diploid_prob_matrix
from state_evolve.tetraploidy_evolution import run_simulation_tetraploidy, tetraploidy_beta_vals, tetraploidy_prob_matrix
from state_evolve.tetraploidy_event import tetraploidy_event, tetraploidy_event_prob
from plot import hist_plot

initial_state = np.array([1, 0, 0])  
mu = 0.02                          
gamma = 0.02                 
time_points = np.linspace(0, 200, 100) 
event_time = 140
evoln_time = 350

def diploid_to_tetraploidy_hist(initial_state, mu, gamma, pre_time, post_time):
    diploid_states = run_simulation_diploid(initial_state, mu, gamma, num_iterations=pre_time)
    beta_vals = []
    beta_vals.append(diploid_beta_vals(diploid_states))
    tetraploidy_initial_state = tetraploidy_event(diploid_states[-1])
    tetraploidy_states = run_simulation_tetraploidy(tetraploidy_initial_state, mu, gamma, num_iterations=post_time)
    beta_vals.append(tetraploidy_beta_vals(tetraploidy_states))

    hist_plot(beta_vals)

def diploid_to_tetraploidy_prob_dist(initial_state, mu, gamma, event_time, evoln_time):

    diploid_evoln_time = np.linspace(0,event_time)
    diploid_probs = diploid_prob_matrix(initial_state, mu, gamma, diploid_evoln_time)
    initial_tetraploidy_probs = tetraploidy_event_prob(diploid_probs)

    tetraploidy_evoln_time = np.linspace(0,evoln_time-event_time)
    tetraploidy_probs = tetraploidy_prob_matrix(initial_tetraploidy_probs, mu, gamma, tetraploidy_evoln_time)
    initial_tetraploidy_probs = np.array([tetraploidy_event_prob(diploid_probs)])
 
    for i in range(diploid_probs.shape[1]):
        plt.plot(diploid_evoln_time, diploid_probs[:, i])
 
    for i in range(tetraploidy_probs.shape[1]):
        plt.plot(tetraploidy_evoln_time + event_time, tetraploidy_probs[:, i])

    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.title('Diploid to Tetraploidy Evolution')
    plt.legend()
    plt.grid()
    plt.show()
    

diploid_to_tetraploidy_prob_dist(initial_state, mu, gamma, event_time, evoln_time)