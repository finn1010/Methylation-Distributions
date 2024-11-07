from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

from plot import plot_prob_dist, hist_plot

'''Trisomy Event Simulation'''

from diploid_trisomy_evoln import diploid_to_trisomy_hist, diploid_to_trisomy_prob_dist


initial_state = np.array([1, 0, 0])  
mu = 0.02                          
gamma = 0.02                 
time_points = np.linspace(0, 200, 100) 
early_event_time = 140
early_evoln_time = 350
early_pre_time = 5000
early_post_time = 5000

diploid_to_trisomy_hist(initial_state, mu, gamma, early_pre_time, early_post_time)
diploid_to_trisomy_prob_dist(initial_state, mu, gamma, early_event_time, early_evoln_time)

late_event_time = 140
late_evoln_time = 350
late_pre_time = 5000
late_post_time = 5000

diploid_to_trisomy_hist(initial_state, mu, gamma, late_pre_time, late_post_time)
diploid_to_trisomy_prob_dist(initial_state, mu, gamma, late_event_time, late_evoln_time)




'''Tetraploidy Event Simulation'''

from diploid_tetraploidy_evoln import diploid_to_tetraploidy_hist, diploid_to_tetraploidy_prob_dist

initial_state = np.array([1, 0, 0])  
mu = 0.02                          
gamma = 0.02                 
time_points = np.linspace(0, 200, 100) 
early_event_time = 140
early_evoln_time = 350
early_pre_time = 5000
early_post_time = 5000

diploid_to_tetraploidy_hist(initial_state, mu, gamma, early_pre_time, early_post_time)
diploid_to_tetraploidy_prob_dist(initial_state, mu, gamma, early_event_time, early_evoln_time)

late_event_time = 140
late_evoln_time = 350
late_pre_time = 5000
late_post_time = 5000

diploid_to_cnLOH_hist(initial_state, mu, gamma, late_pre_time, late_post_time)
diploid_to_cnLOH_prob_dist(initial_state, mu, gamma, late_event_time, late_evoln_time)

'''Copy Neutral Loss of Hetrozygosity Event Simulation'''

from diploid_cnLOH_evoln import diploid_to_cnLOH_hist, diploid_to_cnLOH_prob_dist

initial_state = np.array([1, 0, 0])  
mu = 0.02                          
gamma = 0.02                 
time_points = np.linspace(0, 200, 100) 
early_event_time = 140
early_evoln_time = 350
early_pre_time = 5000
early_post_time = 5000

diploid_to_cnLOH_hist(initial_state, mu, gamma, early_pre_time, early_post_time)
diploid_to_cnLOH_prob_dist(initial_state, mu, gamma, early_event_time, early_evoln_time)

late_event_time = 140
late_evoln_time = 350
late_pre_time = 5000
late_post_time = 5000

diploid_to_cnLOH_hist(initial_state, mu, gamma, late_pre_time, late_post_time)
diploid_to_cnLOH_prob_dist(initial_state, mu, gamma, late_event_time, late_evoln_time)



