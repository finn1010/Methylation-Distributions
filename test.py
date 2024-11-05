from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from state_evolve.diploid_evolution import diploid_prob_matrix

initial_state_diploid = np.array([1, 0, 0])  
mu = 0.02                          
gamma = 0.02                 
t_alteration = 100  
time_points = np.linspace(0, 200, 100)  

RateMatrix_diploid = np.array([[-2*gamma, mu, 0], 
                               [2*gamma, -(gamma + mu), 2*mu], 
                               [0, gamma, -2*mu]])

RateMatrix_trisomy = np.array([[-3*gamma, mu, 0, 0], 
                               [3*gamma, -(2*gamma + mu), 2*mu, 0], 
                               [0, 2*gamma, -(gamma + 2*mu), 3*mu], 
                               [0, 0, gamma, -3*mu]])

Probabilities = []
for t in time_points:
    if t < t_alteration:
        prob_diploid = linalg.expm(RateMatrix_diploid * t) @ initial_state_diploid
        prob_padded = np.append(prob_diploid, 0)  # Pad to 4 states
        Probabilities.append(prob_padded)
    else:
        t_relative = t - t_alteration
        final_diploid_state = linalg.expm(RateMatrix_diploid * t_alteration) @ initial_state_diploid
        initial_state_trisomy = np.zeros(4)
        initial_state_trisomy[:3] = final_diploid_state  # Map diploid state to trisomy initial
        prob_trisomy = linalg.expm(RateMatrix_trisomy * t_relative) @ initial_state_trisomy
        Probabilities.append(prob_trisomy)

Probabilities = np.array(Probabilities)


plt.figure(figsize=(10, 6))
for state in range(Probabilities.shape[1]):
    plt.plot(time_points, Probabilities[:, state], label=f'State {state + 1}')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Probability Distribution with Diploid to Trisomy Transition')
plt.legend()
plt.show()
