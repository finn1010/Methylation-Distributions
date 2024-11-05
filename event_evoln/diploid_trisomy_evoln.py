import numpy as np
import matplotlib.pyplot as plt

from state_evolve.diploid_evolution import run_simulation_diploid, diploid_beta_vals, diploid_prob_matrix
from state_evolve.trisomy_evolution import run_simulation_trisomy, trisomy_beta_vals, trisomy_prob_matrix
from state_evolve.trisomy_event import trisomy_event, trisomy_event_prob
from plot import hist_plot, plot_prob_dist
initial_state = np.array([1, 0, 0])  
mu = 0.02                          
gamma = 0.02                 
event_time = 140
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

    # print('dsfdaf', initial_trisomy_probs.shape)
    trisomy_evoln_time = np.linspace(0,evoln_time-event_time)
    trisomy_probs = trisomy_prob_matrix(initial_trisomy_probs, mu, gamma, trisomy_evoln_time)
    initial_trisomy_probs = np.array([trisomy_event_prob(diploid_probs)])
 
    trisomy_probs = np.concatenate((initial_trisomy_probs,trisomy_probs[:-1]))
    for i in range(diploid_probs.shape[1]):
        plt.plot(diploid_evoln_time, diploid_probs[:, i])
 
    # Plot trisomy probabilities
    for i in range(trisomy_probs.shape[1]):
        plt.plot(trisomy_evoln_time + event_time, trisomy_probs[:, i])

    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.title('Diploid and Trisomy Probability Distributions')
    plt.legend()
    plt.grid()
    plt.show()
   
#    plt.figure(figsize=(10, 6))

# Plot diploid probabilities

   
    # time_points = np.linspace(0,evoln_time)



    # print(diploid_probs.shape)
    # print(trisomy_probs)
    # # padded_diploid_probs = np.pad(diploid_probs, ((0, 0), (0, 1)), mode='constant')

    # # # Concatenate probabilities and time points
    # # full_prob_list = np.concatenate((padded_diploid_probs, trisomy_probs), axis=0)
    # # time_points = np.concatenate((diploid_evoln_time, trisomy_evoln_time))

    # # Plot the full probability distribution over time
    # plot_prob_dist(time_points, trisomy_probs)
    # # # Assuming diploid_probs has shape (n, 3) and trisomy_probs has shape (m, 4)
    # # # Pad diploid_probs with a column of zesros to make it (n, 4)
    # # padded_diploid_probs = np.pad(diploid_probs, ((0, 0), (0, 1)), mode='constant')

    # # # Now you can concatenate along axis 0
    # full_prob_list = np.concatenate((padded_diploid_probs, trisomy_probs), axis=0)

    # trisomy_evoln_time = np.linspace(event_time, evoln_time)
    # plot_prob_dist(time_points, full_prob_list)
    # # plot_prob_dist(diploid_evoln_time, trisomy_probs)

diploid_to_trisomy_prob_dist(initial_state, mu, gamma, event_time, evoln_time)