import numpy as np
import matplotlib.pyplot as plt

# Target distribution
def p(x):
    return np.exp(-2 * x)

# Inverse CDF of target distribution
def inverse_cdf(y):
    return -0.5 * np.log(1 - y)

# Proposal distribution
def q(x):
    return 2 - x

# Accept/reject method
def accept_reject(n):
    x_values = []
    while len(x_values) < n:
        x = np.random.uniform(0, 1)
        u = np.random.uniform(0, q(x))
        if u <= p(x):
            x_values.append(x)
    return x_values

# Generate random numbers using inverse transform method
n = 2000
uniform_random_numbers = np.random.uniform(0, 1, n)
inverse_transform_random_numbers = inverse_cdf(uniform_random_numbers)

# Generate random numbers using accept/reject method
accept_reject_random_numbers = accept_reject(n)

# Histogram the sampled RNG in both cases
plt.hist(inverse_transform_random_numbers, bins=30, alpha=0.5, density = True,label='Inverse transform method')
plt.hist(accept_reject_random_numbers, bins=30, alpha=0.5, density = True,label='Accept/reject method')
plt.xlabel('x')
plt.ylabel('Frequency')
plt.legend()
plt.title('Normalized Histogram)')
plt.savefig('q3_normalized_histogram.png')
plt.close()
plt.hist(inverse_transform_random_numbers, bins=30, alpha=0.5,label='Inverse transform method')
plt.hist(accept_reject_random_numbers, bins=30, alpha=0.5,label='Accept/reject method')
plt.xlabel('x')
plt.ylabel('Frequency')
plt.legend()
plt.title('Histogram')
plt.savefig('q3_histogram.png')
plt.close()
