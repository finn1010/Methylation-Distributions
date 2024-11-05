import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

gamma = 0.02
mu = 0.02

def transition_matrix(gamma, mu):
    row1 = [-2*gamma, mu, 0]
    row2 = [2*gamma, -(gamma + mu), 2*mu]
    row3 = [0, gamma, -2*mu]
    T = np.array([row1, row2, row3])
    return T

def state_matrix(m, k, u):
    S = np.array([m, k, u])
    return S

def ode_fn(T):
    def ode(t, S):
        return T.dot(S)
    return ode

def main():
    S = state_matrix(1, 0, 0)  
    time_points = np.linspace(0, 200, 100)
    T = transition_matrix(gamma, mu)
    ode = ode_fn(T)

    solution = solve_ivp(ode, [time_points[0], time_points[-1]], S, t_eval=time_points)

    plt.figure(figsize=(10, 6))
    plt.plot(solution.t, solution.y[0], label='m (State 1)', color='blue')
    plt.plot(solution.t, solution.y[1], label='k (State 2)', color='orange')
    plt.plot(solution.t, solution.y[2], label='u (State 3)', color='green')
    
#convergence happens because the transition matrix describes
#a stochastic process that stabilizes over time into a stationary distribution
    plt.xlabel('Time')
    plt.ylabel('State Variables')
    plt.legend()
    plt.grid()
    plt.show()
    #plot shows that over time there is a 25

if __name__ == "__main__":
    main()
