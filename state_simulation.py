import numpy as np
import scipy
from state_probabilities import transition_matrix
import matplotlib.pyplot as plt

def state_simulation(mu, gamma):
    rng = np.random.default_rng()
    m=0
    k=1
    dt=1
    w=0
    p_m_to_k = 2 * gamma * dt if m == 1 else 0  
    p_k_to_m = gamma * dt if k == 1 else 0         
    p_k_to_w = mu * dt if k == 1 else 0      
    p_w_to_k = 2 * mu * dt if w == 1 else 0    

    if m == 1:
        if rng.random() < p_m_to_k:
            m, k, w = 0, 1, 0  

    elif k == 1:
        if rng.random() < p_k_to_m:
            m, k, w = 1, 0, 0  
            
        elif rng.random() < p_k_to_w:
            m, k, w = 0, 0, 1  

    elif w == 1:
        if rng.random() < p_w_to_k:
            m, k, w = 0, 1, 0  

    return [m, k, w]

x = []

def run_simulation():
    mu=0.5
    gamma=0.5
    x.append(state_simulation(mu,gamma))
    return x

for i in range(5):
    run_simulation()

m_list = []
m_vals = [subarray[0] for subarray in x]
m_list.append([])

k_list = []
k_vals = [subarray[1] for subarray in x]
k_list.append([])

w_list = []
w_vals = [subarray[2] for subarray in x]
w_list.append([])

for i in range(4000): 
    run_simulation()

m_vals = [subarray[0] for subarray in x]
k_vals = [subarray[1] for subarray in x]

beta_vals = [(k + 2*m) / 2 for m, k in zip(m_vals, k_vals)]

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(beta_vals, bins=30, edgecolor='black')
plt.title('Histogram of Beta Values')
plt.xlabel('Beta')
plt.ylabel('Frequency')
plt.show()
