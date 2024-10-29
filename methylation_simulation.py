import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def transition_matrix(gamma, mu):
    row1 = [-2*gamma, mu, 0]
    row2 = [2*gamma, -(gamma + mu), 2*mu]
    row3 = [0, gamma, -2*mu]
    T = np.array([row1, row2, row3])
    return T

def state_matrix(m, k, u):
    S = np.array([m, k, u])
    return S

def create_ODE(T):
    def ode(t, S):
        return T.dot(S)
    return ode

def main():
    gamma = 0.2
    mu = 0.2
    T = transition_matrix(gamma, mu)
    S = state_matrix(0, 0, 1)  

    ode_func = create_ODE(T)

    time_points = np.linspace(0, 10, 10)

    solution = solve_ivp(ode_func, [time_points[0], time_points[-1]], S, t_eval=time_points)

    plt.figure(figsize=(10, 6))
    plt.plot(solution.t, solution.y[0], label='m (State 1)', color='blue')
    plt.plot(solution.t, solution.y[1], label='k (State 2)', color='orange')
    plt.plot(solution.t, solution.y[2], label='u (State 3)', color='green')
    
    plt.xlabel('Time')
    plt.ylabel('State Variables')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
