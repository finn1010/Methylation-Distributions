import numpy as np
from state_evolve.diploid_evolution import run_simulation_diploid, diploid_beta_vals
from state_evolve.trisomy_evolution import run_simulation_trisomy, trisomy_beta_vals
from state_evolve.trisomy_event import trisomy_event
from plot import hist_plot
initial_state = np.array([1, 0, 0])  
mu = 0.02                          
gamma = 0.02                 
time_points = np.linspace(0, 200, 100) 
diploid_states = run_simulation_diploid(initial_state, mu, gamma, num_iterations=50000)
beta_vals = []
beta_vals.append(diploid_beta_vals(diploid_states))
trisomy_initial_state = trisomy_event(diploid_states[-1])
trisomy_states = run_simulation_trisomy(trisomy_initial_state, mu, gamma, num_iterations=50000)
beta_vals.append(trisomy_beta_vals(trisomy_states))

hist_plot(beta_vals)