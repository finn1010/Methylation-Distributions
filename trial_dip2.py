from scipy import linalg, stats
import numpy as np

def state_simulation(initial_state, mu, gamma, t):

    R = np.array([
        [-2 * gamma, mu, 0],
        [2 * gamma, -(gamma + mu), 2 * mu],
        [0, gamma, -2 * mu]
    ])
    initial_state_array = np.array(initial_state)
    state_probs = linalg.expm(R * t) @ initial_state_array

    return state_probs


def cnloh_event(probabilities):
    m_prob, k_prob, w_prob = probabilities
    m_prob = m_prob + 0.5 * k_prob
    w_prob = w_prob + 0.5 * k_prob
    k_prob = 0
    initial_cnLOH_probs = [m_prob, k_prob, w_prob]
    return initial_cnLOH_probs 

def calculate_individual_beta_vals(states):
    m, k, w = states

    beta_vals = np.array([1.0] * m + [0.5] * k + [0.0] * w)

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
    beta = kappa * (1- mu)

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

initial_state = [0.5,0,0.5]
mu, gamma = 0.02, 0.02
t = 2
num_sites=10000



def cnloh_sim(initial_state, mu, gamma, patient_age, event_time, num_sites):
    states = state_simulation(initial_state, mu, gamma, event_time)
    print(states)
    states = cnloh_event(states)
    states = state_simulation(states,mu,gamma,patient_age - event_time)
    sampled_state = np.random.multinomial(num_sites, states)
    sampled_state = sampled_state.tolist()
    beta_vals = calculate_individual_beta_vals(sampled_state)
    noisy_beta_vals = add_noise(beta_vals, 0.05,0.95,30)

    return noisy_beta_vals

# patient_age = 60
# event_time = 10
# mu, gamma = 0.02, 0.02
# noisy_beta = cnloh_sim(initial_state, mu, gamma, patient_age, event_time, num_sites)
# print(noisy_beta)

# from plot import hist_plot

# hist_plot(noisy_beta, noisy_beta,'Diploid', event_time, patient_age-event_time, 'e.png')


# from plot import hist_plot
# mu, gamma = 0.02, 0.02
# patient_age = 60
# event_time = 50
# noisy_beta = cnloh_sim(initial_state, mu, gamma, patient_age, event_time, num_sites)
# hist_plot(noisy_beta, noisy_beta,'Diploid', event_time, patient_age-event_time, 'e.png')
