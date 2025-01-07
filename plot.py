from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from colours import pallet_dip,pallet_tri

def plot_prob_dist(time_points, probabilities, plt_title):
    plt.figure(figsize=(10, 6))
    for state in range(probabilities.shape[1]):
        plt.plot(time_points, probabilities[:, state], label=f'State {state + 1}')
    plt.xlabel('Time (years)')  
    plt.ylabel('Probability')
    plt.title(f'{plt_title}')
    plt.legend()
    plt.show()

def hist_plot(beta_vals_before, beta_vals_after, event_type, pre_time,post_time,fig_name):
    # plt.hist(beta_vals_before, bins=30, edgecolor='black', alpha=0.5, label=f'Time before event ({pre_time} years)', density=False, align='mid')
    plt.hist(beta_vals_after, bins=30, edgecolor='black', alpha=0.5, label=f'Time after event ({post_time} years)', density=False, align='mid')
    
    plt.xlabel('Beta Values')
    plt.ylabel('Probability Density')
    plt.title(f'Beta values before and after Diploid to {event_type} event')
    plt.legend(loc='upper right', fontsize=8)
    plt.savefig(f'{fig_name}.pdf', format='pdf', dpi=300)
    plt.show()

# def hist_plot(beta_vals):
#     plt.figure(figsize=(10, 6))
#     plt.hist(beta_vals, bins=30, edgecolor='black')
#     plt.title('Histogram of Beta Values')
#     plt.xlabel('Beta')
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.show()