import numpy as np

def trisomy_event(mkw):
    rng = np.random.default_rng()  
    m_cancer, k_cancer, w_cancer = mkw
    u_cancer = rng.binomial(n = k_cancer, p = 0.5)
    d_cancer = k_cancer - u_cancer

    mudw = np.array([m_cancer, u_cancer, d_cancer, w_cancer])

    return mudw