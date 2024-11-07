from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

from plot import plot_prob_dist, hist_plot

'''Diploid Evolution Simulation'''

from state_evolve.diploid_evolution import diploid_prob_matrix, diploid_simulation

initial_state = np.array([1, 0, 0])  
mu = 0.02                  
gamma = 0.02         
time_points = np.linspace(0, 200, 100)   

dip_prob_matrix = diploid_prob_matrix(initial_state, mu, gamma, time_points)
plot_prob_dist(time_points, dip_prob_matrix)

beta_vals = diploid_simulation(initial_state, mu, gamma)
hist_plot(beta_vals)

'''Trisomy Evolution Simulation'''

from state_evolve.trisomy_evolution import trisomy_prob_matrix, trisomy_simulation

initial_state = np.array([1, 0, 0, 0])  
mu = 0.02                  
gamma = 0.02         
time_points = np.linspace(0, 200, 100)   

tri_prob_matrix = trisomy_prob_matrix(initial_state, mu, gamma, time_points)
plot_prob_dist(time_points, tri_prob_matrix)

beta_vals = trisomy_simulation(initial_state, mu, gamma)
hist_plot(beta_vals)



from state_evolve.tetraploidy_evolution import tetraploidy_prob_matrix, tetraploidy_simulation

'''Tetraploidy Evolution Simulation'''

initial_state = np.array([1, 0, 0, 0, 0])  
mu = 0.02                  
gamma = 0.02         
time_points = np.linspace(0, 200, 100)   

tet_prob_matrix = tetraploidy_prob_matrix(initial_state, mu, gamma, time_points)
plot_prob_dist(time_points, tet_prob_matrix)

beta_vals = tetraploidy_simulation(initial_state, mu, gamma)
hist_plot(beta_vals)
