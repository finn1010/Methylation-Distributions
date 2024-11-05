import numpy as np
from state_evolve.diploid_evolution import run_simulation_diploid, diploid_beta_vals
from state_evolve.tetraploidy_evolution import run_simulation_tetraploidy, tetraploidy_beta_vals
from state_evolve.tetraploidy_event import tetraploidy_event
from plot import hist_plot
initial_state = np.array([1, 0, 0])  
mu = 0.02                          
gamma = 0.02                 
time_points = np.linspace(0, 200, 100) 
def diploid_to_trisomy_hist(initial_state, mu, gamma):
    diploid_states = run_simulation_diploid(initial_state, mu, gamma, num_iterations=50000)
    beta_vals = []
    beta_vals.append(diploid_beta_vals(diploid_states))
    tetraploidy_initial_state = tetraploidy_event(diploid_states[-1])
    tetraploidy_states = run_simulation_tetraploidy(tetraploidy_initial_state, mu, gamma, num_iterations=50000)
    beta_vals.append(tetraploidy_beta_vals(tetraploidy_states))

    hist_plot(beta_vals)