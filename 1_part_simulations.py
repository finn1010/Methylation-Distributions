from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

from plot import plot_prob_dist, hist_plot

'''Diploid Evolution Simulation'''

from state_evolve.diploid_evolution import diploid_prob_matrix, run_simulation_diploid, diploid_beta_vals

# initial_state = np.array([1, 0, 0])  
# mu = 0.02                  
# gamma = 0.02         
# time_points = np.linspace(0, 200, 100)   

# dip_prob_matrix = diploid_prob_matrix(initial_state, mu, gamma, time_points)
# plot_prob_dist(time_points, dip_prob_matrix)

# final_states = run_simulation_diploid(mu, gamma, num_cells = 100, num_iterations=10)
# beta_vals = diploid_beta_vals(final_states)
# hist_plot(beta_vals)

'''Trisomy Evolution Simulation'''

from state_evolve.trisomy_evolution import trisomy_prob_matrix, run_simulation_trisomy, trisomy_beta_vals

initial_state = np.array([1, 0, 0, 0])  
mu = 0.02                  
gamma = 0.02         
time_points = np.linspace(0, 200, 100)   

tri_prob_matrix = trisomy_prob_matrix(initial_state, mu, gamma, time_points)
plot_prob_dist(time_points, tri_prob_matrix)

final_states = run_simulation_trisomy(mu, gamma, num_cells=100, num_iterations=4)
beta_vals = trisomy_beta_vals(final_states)
hist_plot(beta_vals)


from state_evolve.tetraploidy_evolution import tetraploidy_prob_matrix, run_simulation_tetraploidy, tetraploidy_beta_vals

'''Tetraploidy Evolution Simulation'''

# initial_state = np.array([1, 0, 0, 0, 0])  
# mu = 0.02                  
# gamma = 0.02         
# time_points = np.linspace(0, 200, 100)   

# tet_prob_matrix = tetraploidy_prob_matrix(initial_state, mu, gamma, time_points)
# plot_prob_dist(time_points, tet_prob_matrix)

# final_states = run_simulation_tetraploidy(mu, gamma,num_cells=100, num_iterations=10)
# beta_vals = tetraploidy_beta_vals(final_states)
# hist_plot(beta_vals)
