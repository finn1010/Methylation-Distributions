from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

def plot_prob_dist(time_points, probabilities):
    plt.figure(figsize=(10, 6))
    for state in range(probabilities.shape[1]):
        plt.plot(time_points, probabilities[:, state], label=f'State {state + 1}')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    # plt.title('Probability Distribution of Methylation States Over Time')
    plt.legend()
    plt.show()

def hist_plot(beta_vals):
    plt.figure(figsize=(10, 6))
    plt.hist(beta_vals, bins=30, edgecolor='black')
    plt.title('Histogram of Beta Values')
    plt.xlabel('Beta')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()