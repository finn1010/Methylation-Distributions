from scipy import linalg, stats
import numpy as np

def diploid_simulation(initial_state, mu, gamma, t, num_sites):

    R = np.array([
        [-2 * gamma, mu, 0],
        [2 * gamma, -(gamma + mu), 2 * mu],
        [0, gamma, -2 * mu]
    ])

    initial_state_array = np.array(initial_state)
    state_probs = linalg.expm(R * t) @ initial_state_array

    return state_probs

def trisomy_simulation(initial_state, mu, gamma, t, num_sites):
    
    R = np.array([[-3*mu, gamma, 0, 0], 
                    [3*mu, -(gamma+2*mu), 2*gamma, 0], 
                    [0, 2*mu, -(2*gamma+mu), 3*gamma],
                    [0, 0, mu, -3*gamma]])

    initial_state_array = np.array(initial_state)
    state_probs = linalg.expm(R * t) @ initial_state_array

    return state_probs


def trisomy_event(probabilities):
    m_prob, k_prob, w_prob = probabilities
    u_prob = d_prob = k_prob / 2
    initial_trisomy_probs = [m_prob, u_prob, d_prob, w_prob]
    return initial_trisomy_probs 

def beta_calc(states):
    m, u, d, w = states

    m1, u1, d1, w1 = states
    beta_vals = np.array([1.0] * m1 + [2/3] * u1 + [1/3] * d1 + [0.0] * w1)

    return beta_vals

def beta_convert_params(mu, kappa):
    """
    Convert mean/dispersion parameterization of a beta distribution to the ones
    scipy supports
    """

    if np.any(kappa <= 0):
        raise Exception("kappa must be greater than 0")
    elif np.any(mu <= 0) or np.any(mu >= 1):
        raise Exception("mu must be between 0 and 1")
    
    alpha = kappa * mu 
    beta = kappa * (1 - mu)

    return alpha, beta

def beta_rvs(mean, kappa, **kwargs):
    """
    Generate random samples from a beta distribution with mean/dispersion
    specified
    """
    alpha, beta = beta_convert_params(mean, kappa)

    return stats.beta.rvs(alpha, beta, **kwargs)

def rescale_beta(beta, delta, eta):
    """
    Linear transform of beta values from between 0 and 1 to between delta and 
    eta
    """
    return (eta - delta) * beta + delta

def add_noise(beta, delta, eta, kappa):
    """
    Rescale distribution to lie between delta and eta and add beta distributed 
    noise
    """
    beta_rescale = rescale_beta(beta, delta, eta)
 
    return beta_rvs(beta_rescale, kappa)



def trisomy_sim(initial_state, mu, gamma, patient_age, event_time, num_sites):
    states = diploid_simulation(initial_state, mu, gamma, event_time, num_sites)
    states = trisomy_event(states)
    states = trisomy_simulation(states,mu,gamma,patient_age - event_time,num_sites)
    sampled_state = np.random.multinomial(num_sites, states)
    sampled_state = sampled_state.tolist()
    beta_vals = beta_calc(sampled_state)
    noisy_beta_vals = add_noise(beta_vals, 0.05,0.95,30)

    return noisy_beta_vals

# initial_state = [0.5,0,0.5]
# num_sites = 10000
# patient_age = 60
# event_time = 10
# mu, gamma = 0.02, 0.02
# noisy_beta = trisomy_sim(initial_state, mu, gamma, patient_age, event_time, num_sites)
# # print(noisy_beta)

# from plot import hist_plot

# hist_plot(noisy_beta, noisy_beta,'Diploid', event_time, patient_age-event_time, 'e.png')


# from plot import hist_plot
# mu, gamma = 0.02, 0.02
# patient_age = 60
# event_time = 50
# noisy_beta = trisomy_sim(initial_state, mu, gamma, patient_age, event_time, num_sites)
# hist_plot(noisy_beta, noisy_beta,'Diploid', event_time, patient_age-event_time, 'e.png')
