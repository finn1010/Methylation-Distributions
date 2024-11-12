from scipy import linalg
import numpy as np

def state_initialisation():
    rng = np.random.default_rng()
    m = [1,0,0,0,0]
    w = [0,0,0,0,1]
    if rng.random() > 0.5:
        state = m
    else:
        state = w
    return state

def tetraploidy_prob_matrix(initial_state, mu, gamma, time_points):
    RateMatrix = np.array([
        [-4*mu, gamma, 0, 0, 0],                
        [4*mu, -(3 * mu + gamma), 2 * gamma, 0, 0], 
        [0, 3 * mu, -(2 * mu + 2*gamma), 3*gamma, 0], 
        [0, 0, 2*mu, -(3*gamma+mu), 4*gamma],
        [0, 0, 0, mu, -4*gamma]           
    ])

    Probabilities = np.zeros((len(time_points), len(initial_state)))
    for i, t in enumerate(time_points):
        ProbStates = linalg.expm(RateMatrix * t) @ initial_state
        Probabilities[i] = ProbStates / np.sum(ProbStates) 

    return Probabilities

def calc_dt_trisomy(mu, gamma):
    dt_max = 0.1 / np.max((
    4*mu, 
    4*gamma)
    )
    return dt_max 

def state_simulation(initial_state, mu, gamma):
    rng = np.random.default_rng()
    m, k, d, v, w = initial_state
    dt=1
    p_m_to_k = 4*mu * dt if m == 1 else 0  
    p_k_to_m = gamma * dt if k == 1 else 0         
    p_k_to_d = 3*mu * dt if k == 1 else 0      
    p_d_to_k = 2*gamma * dt if d == 1 else 0    
    p_d_to_v = 2*mu * dt if d == 1 else 0      
    p_v_to_d = 3*gamma * dt if v == 1 else 0    
    p_v_to_w = mu * dt if v == 1 else 0      
    p_w_to_v = 4*gamma * dt if w == 1 else 0      

    if m == 1:
        if rng.random() < p_m_to_k:
            m, k, d, v, w = 0, 1, 0, 0, 0

    elif k == 1:
        if rng.random() < p_k_to_m:
            m, k, d, v, w = 1, 0, 0, 0, 0
            
        elif rng.random() < p_k_to_d:
            m, k, d, v, w = 0, 0, 1, 0, 0

    elif d == 1:
        if rng.random() < p_d_to_k:
            m, k, d, v, w = 0, 1, 0, 0, 0
            
        elif rng.random() < p_d_to_v:
            m, k, d, v, w = 0, 0, 0, 1, 0
    
    elif v == 1:
        if rng.random() < p_v_to_d:
            m, k, d, v, w = 0, 0, 1, 0, 0
            
        elif rng.random() < p_v_to_w:
            m, k, d, v, w = 0, 0, 0, 0, 1

    elif w == 1:
        if rng.random() < p_w_to_v:
            m, k, d, v, w = 0, 0, 0, 1, 0

    return [m, k, d, v, w]

def run_simulation_tetraploidy(mu, gamma, initial_state, num_cells=100, num_iterations=10):
    states = []
    final_states = []
    dt_max = calc_dt_trisomy(mu, gamma)
    current_state = initial_state
    for _ in range(num_iterations):
        current_state = state_simulation(current_state, mu, gamma)
        states.append(current_state)
    final_states.append(states[-1])
    
    return final_states


def tetraploidy_beta_vals(states):
    beta_vals = [(state[1] + 2 * state[2] + 3 * state[3] + 4 * state[4]) / 4 for state in states]
    return beta_vals