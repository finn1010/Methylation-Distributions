from scipy import linalg
import numpy as np

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

def diploid_prob_matrix(initial_state, mu, gamma, t):
    RateMatrix = np.array([[-2*gamma, mu, 0], 
                                [2*gamma, -(gamma + mu), 2*mu], 
                                [0, gamma, -2*mu]])

    Probabilities = np.array(linalg.expm(RateMatrix * t) @ initial_state / np.sum(initial_state))
                             
    return Probabilities

def tetraploidy_prob_matrix(initial_state, mu, gamma, t):
    RateMatrix = np.array([
        [-4*mu, gamma, 0, 0, 0],                
        [4*mu, -(3 * mu + gamma), 2 * gamma, 0, 0], 
        [0, 3 * mu, -(2 * mu + 2*gamma), 3*gamma, 0], 
        [0, 0, 2*mu, -(3*gamma+mu), 4*gamma],
        [0, 0, 0, mu, -4*gamma]           
    ])

    Probabilities = np.array(linalg.expm(RateMatrix * t) @ initial_state / np.sum(initial_state))

    return Probabilities

def tetraploidy_event_prob(probabilities):
    m_prob,d_prob, w_prob = probabilities
    k_prob = v_prob = 0
    initial_tetraploidy_probs = [m_prob, k_prob, d_prob, v_prob, w_prob]
    return initial_tetraploidy_probs 

def tet_sim(initial_state,mu, gamma, event_time, patient_age):
    probs = diploid_prob_matrix(initial_state, mu, gamma, event_time)
    probs = tetraploidy_event_prob(probs)
    probs = tetraploidy_prob_matrix(probs, mu, gamma, patient_age-event_time)
