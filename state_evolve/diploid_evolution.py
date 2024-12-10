from scipy import linalg
import numpy as np

def state_initialisation():
    rng = np.random.default_rng()
    m = [1,0,0]
    w = [0,0,1]
    if rng.random() > 0.5:
        state = m
    else:
        state = w
    return state

def ss_initialisation(mu, gamma):
    prob = [
    gamma ** 2 / (mu + gamma) ** 2, 
    2 * mu * gamma / (mu + gamma) ** 2, 
    mu ** 2 / (mu + gamma) ** 2
    ]

    states = [
    [1, 0, 0], 
    [0, 1, 0],  
    [0, 0, 1]   
    ]
    
    rng = np.random.default_rng()
    state = rng.choice(states, p=prob)
    
    return state

def ss_init_prob(mu,gamma):
    return [
    gamma ** 2 / (mu + gamma) ** 2, 
    2 * mu * gamma / (mu + gamma) ** 2, 
    mu ** 2 / (mu + gamma) ** 2
    ]



def diploid_prob_matrix(initial_state, mu, gamma, time_points):
    RateMatrix = np.array([[-2*gamma, mu, 0], 
                                [2*gamma, -(gamma + mu), 2*mu], 
                                [0, gamma, -2*mu]])

    Probabilities = np.array([linalg.expm(RateMatrix * t) @ initial_state / np.sum(initial_state) 
                            for t in time_points])
    return Probabilities


def calc_dt_max_diploid(mu, gamma):
    dt_max = 0.1 / np.max((
    2*mu, 
    2*gamma)
    )
    return dt_max 

def diploid_dt(start_evoln, end_evoln, dt_max):
    n = int((end_evoln-start_evoln) / dt_max) + 2  
    t = np.linspace(start_evoln, end_evoln, n) 
    dt = t[1] - t[0]
    return dt


def state_simulation(initial_state, mu, gamma, dt):
    rng = np.random.default_rng()
    m,k,w = initial_state
    
    p_m_to_k = 2 * gamma * dt if m == 1 else 0  
    p_k_to_m = gamma * dt if k == 1 else 0         
    p_k_to_w = mu * dt if k == 1 else 0      
    p_w_to_k = 2 * mu * dt if w == 1 else 0    

    if m == 1:
        if rng.random() < p_m_to_k:
            m, k, w = 0, 1, 0  

    elif k == 1:
        rand_val = rng.random()
        if rand_val < p_k_to_m:
            m, k, w = 1, 0, 0  
            
        elif rand_val < p_k_to_w + p_k_to_m:
            m, k, w = 0, 0, 1  

        else:
            m, k, w = 0, 1, 0

    elif w == 1:
        if rng.random() < p_w_to_k:
            m, k, w = 0, 1, 0  

    return [m, k, w]


# def run_simulation_diploid(mu, gamma, num_sites=100, start_evoln=0, end_evoln=10, initial_state=None):
#     states = []
#     final_states = []
#     dt_max = calc_dt_max_diploid(mu, gamma)
#     dt = diploid_dt(start_evoln, end_evoln, dt_max)
    
#     if initial_state is None:
#         for _ in range(num_sites):
#             current_state = state_initialisation()
            
#             for _ in range(int(end_evoln/dt)+1):
#                 current_state = state_simulation(current_state, mu, gamma, dt)
#                 states.append(current_state)
#             final_states.append(states[-1])
#     else:
#         current_state = initial_state
#         for _ in range(int(end_evoln/dt)+1):
#             current_state = state_simulation(current_state, mu, gamma,dt)
#             states.append(current_state)
#         final_states.append(states[-1])
    
#     return final_states

def run_simulation_diploid(mu, gamma, init_fn, num_sites=100, start_evoln=0, end_evoln=10, initial_state=None):
    """
    Simulates the evolution of diploid states over time.

    Args:
        mu (float): Mutation rate.
        gamma (float): Conversion rate.
        init_fn (callable): Function to initialize states.
        num_sites (int): Number of sites to simulate (default=100).
        start_evoln (float): Start time of evolution (default=0).
        end_evoln (float): End time of evolution (default=10).
        initial_state (list): Initial state for the simulation (default=None).

    Returns:
        list: Final states of the simulation.
    """
    states = []
    final_states = []
    dt_max = calc_dt_max_diploid(mu, gamma)
    dt = diploid_dt(start_evoln, end_evoln, dt_max)

    if initial_state is None:
        for _ in range(num_sites):
            current_state = init_fn(mu, gamma) if init_fn.__code__.co_argcount > 0 else init_fn()
            for _ in range(int(end_evoln / dt) + 1):
                current_state = state_simulation(current_state, mu, gamma, dt)
                states.append(current_state)
            final_states.append(states[-1])
    else:
        current_state = initial_state
        for _ in range(int(end_evoln / dt) + 1):
            current_state = state_simulation(current_state, mu, gamma, dt)
            states.append(current_state)
        final_states.append(states[-1])

    return final_states

def diploid_beta_vals(states):
    beta_vals = [(state[1] + 2 * state[0]) / 2 for state in states]
    return beta_vals

def cnLOH_event(mkw):
    rng = np.random.default_rng()  
    m_cancer, k_cancer, w_cancer = mkw
    if k_cancer == 1:
        if rng.random() < 0.5:
            m_cancer += k_cancer
            k_cancer = 0
        else:
            w_cancer += k_cancer
            k_cancer = 0
    
    return np.array([m_cancer, k_cancer, w_cancer])
