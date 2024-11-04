from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

initial_state = np.array([1, 0, 0, 0])  
mu = 0.02                          
gamma = 0.02                 
time_points = np.linspace(0, 200, 100)  


RateMatrix = np.array([[-3*mu, gamma, 0, 0], 
                            [3*mu, -(gamma+2*mu), 2*gamma, 0], 
                            [0, 2*mu, -(2*gamma+mu), 3*gamma],
                            [0, 0, mu, -3*gamma]])

Probabilities = np.zeros((len(time_points), len(initial_state)))
for i, t in enumerate(time_points):
    ProbStates = linalg.expm(RateMatrix * t) @ initial_state
    Probabilities[i] = ProbStates / np.sum(ProbStates) 

plt.figure(figsize=(10, 6))
for state in range(4):
    plt.plot(time_points, Probabilities[:, state], label=f'State {state + 1}')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Probability Distribution of Methylation States Over Time')
plt.legend()
plt.show()

def state_simulation(initial_state, mu, gamma):
    rng = np.random.default_rng()
    m,k,d,w = initial_state
    dt=1

    p_m_to_k = 3*mu * dt if m == 1 else 0  
    p_k_to_m = gamma * dt if k == 1 else 0         
    p_k_to_d = 2*mu * dt if k == 1 else 0      
    p_d_to_k = 2*gamma * dt if d == 1 else 0    
    p_d_to_w = mu * dt if d == 1 else 0      
    p_w_to_d = 3*gamma * dt if w == 1 else 0    

    if m == 1:
        if rng.random() < p_m_to_k:
            m, k, d, w = 0, 1, 0, 0 

    elif k == 1:
        if rng.random() < p_k_to_m:
            m, k, d, w = 1, 0, 0, 0
            
        elif rng.random() < p_k_to_d:
            m, k, d, w = 0, 0, 1, 0

    elif d == 1:
        if rng.random() < p_d_to_k:
            m, k, d, w = 0, 1, 0, 0
            
        elif rng.random() < p_d_to_w:
            m, k, d, w = 0, 0, 0, 1

    elif w == 1:
        if rng.random() < p_w_to_d:
            m, k, d, w = 0, 0, 1, 0

    return [m, k, d, w]

x = []

def run_simulation(initial_state, mu, gamma, num_iterations=5000):
    current_state = initial_state
    for _ in range(num_iterations):
        current_state = state_simulation(current_state, mu, gamma)  
        x.append(current_state)  

    return x

final_states = run_simulation(initial_state, mu, gamma)

beta_vals = [(state[1] + 2 * state[2] + 3 * state[3]) / 3 for state in final_states]

# plt.figure(figsize=(10, 6))
# plt.hist(beta_vals, bins=30, edgecolor='black')
# plt.title('Histogram of Beta Values')
# plt.xlabel('Beta')
# plt.ylabel('Frequency')
# plt.show()