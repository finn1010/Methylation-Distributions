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


def diploid_to_tetraploidy_hist(mu, gamma, num_cells, pre_time, post_time):
    final_diploid_states = run_simulation_diploid(mu, gamma, num_cells, pre_time,initial_state=None)
    beta_vals = []
    beta_vals.append(diploid_beta_vals(final_diploid_states))
    tetraploidy_state_list = []
    for state in final_diploid_states:
        tetraploidy_initial_state = tetraploidy_event(state)
        tetraploidy_states = run_simulation_tetraploidy(mu, gamma,tetraploidy_initial_state, num_cells, post_time)
        tetraploidy_state_list.extend(tetraploidy_states)
    print(tetraploidy_state_list)
    beta_vals.append(tetraploidy_beta_vals(tetraploidy_state_list))
 
    hist_plot(beta_vals)

diploid_to_tetraploidy_hist(mu, gamma, 100, 10, 50)
diploid_to_tetraploidy_hist(mu, gamma, 100, 50, 10)


def diploid_to_tetraploidy_prob_dist(initial_state, mu, gamma, event_time, evoln_time):

    diploid_evoln_time = np.linspace(0,event_time)
    diploid_probs = diploid_prob_matrix(initial_state, mu, gamma, diploid_evoln_time)
    initial_tetraploidy_probs = tetraploidy_event_prob(diploid_probs)

    tetraploidy_evoln_time = np.linspace(0,evoln_time-event_time)
    tetraploidy_probs = tetraploidy_prob_matrix(initial_tetraploidy_probs, mu, gamma, tetraploidy_evoln_time)
    initial_tetraploidy_probs = np.array([tetraploidy_event_prob(diploid_probs)])
    
    methylated_dip = [2,1,0]
    for i in range(diploid_probs.shape[1]):
        plt.plot(diploid_evoln_time, diploid_probs[:, i], label=f'{methylated_dip[i]}')

    # methylated_tri = []
    for i in range(tetraploidy_probs.shape[1]):
        plt.plot(tetraploidy_evoln_time + event_time, tetraploidy_probs[:, i], label=f'Tetraploidy State {i+1}')

    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.title('Diploid to Tetraploidy Evolution')
    plt.legend(title='Number of methylated alleles')
    plt.grid()
    plt.show()
    

event_time = 10
evoln_time = 60
diploid_to_tetraploidy_prob_dist(initial_state, mu, gamma, event_time, evoln_time)
event_time = 50
evoln_time = 60
diploid_to_tetraploidy_prob_dist(initial_state, mu, gamma, event_time, evoln_time)