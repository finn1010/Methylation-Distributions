from scipy import linalg
import numpy as np

initial_state = np.array([1, 0, 0])  
mu = 0.02                          
gamma = 0.02                 
time_points = np.linspace(0, 200, 100)  

def diploid_prob_matrix(initial_state, mu, gamma, time_points):
    RateMatrix = np.array([[-2*gamma, mu, 0], 
                                [2*gamma, -(gamma + mu), 2*mu], 
                                [0, gamma, -2*mu]])

    Probabilities = np.array([linalg.expm(RateMatrix * t) @ initial_state / np.sum(initial_state) 
                            for t in time_points])
    return Probabilities

s = diploid_prob_matrix(initial_state, mu, gamma, time_points)

def state_simulation(initial_state, mu, gamma):
    rng = np.random.default_rng()
    m,k,w = initial_state
    dt=1

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

def run_simulation_diploid(initial_state, mu, gamma, num_iterations=5000):
    states = []
    current_state = initial_state
    for _ in range(num_iterations):
        current_state = state_simulation(current_state, mu, gamma)
        states.append(current_state)
    return states

def diploid_simulation(initial_state, mu, gamma):
    final_states = run_simulation_diploid(initial_state, mu, gamma)
    beta_vals = [(state[1] + 2 * state[0]) / 2 for state in final_states]
    return beta_vals

def diploid_beta_vals(states):
    beta_vals = [(state[1] + 2 * state[0]) / 2 for state in states]
    return beta_vals  

