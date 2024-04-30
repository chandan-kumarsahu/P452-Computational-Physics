import matplotlib.pyplot as plt
import numpy as np


def Monte_Carlo_Cos_Integral_mlcp(seed,a, m, N, step_size):
    seed = 1
    results = []
    N_values = list(range(1000, N + 1, step_size))

    for N in N_values:
        x = seed
        sum_cos = 0
        for _ in range(N):
            x = (a * x) % m
            x_scaled = (x / m) * np.pi - np.pi / 2  
            sum_cos += np.cos(x_scaled)
        integral_approx = (np.pi) * sum_cos / N
        results.append(integral_approx)
    
    return N_values, results

N = 100000
step_size = 1000
seed=100
# Using both sets of parameters
N_values, integral_set_65 = Monte_Carlo_Cos_Integral_mlcp(seed,a=65, m=1021, N=N, step_size=step_size)
# _, integral_set_572 = Monte_Carlo_Cos_Integral_mlcp(seed,a=572, m=16381, N=N, step_size=step_size)

# Plotting the results
plt.figure(figsize=(12, 8))
plt.plot(N_values, integral_set_65, label='a = 65, m = 1021')
# plt.plot(N_values, integral_set_572, label='a = 572, m = 16381')
# plt.axhline(y=2, color='r', linestyle='--', label='True value (2)')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Approximated Integral')
plt.title('Convergence of Monte Carlo Approximation of the Integral of cos(x)')
plt.legend()
plt.grid(True)
plt.show()