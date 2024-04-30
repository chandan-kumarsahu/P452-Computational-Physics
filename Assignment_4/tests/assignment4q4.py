import numpy as np

# Function to integrate
def f(x):
    return np.exp(-2 * x) / (1 + x**2)

# Importance sampling functions
def p1(x):
    return 0.5

def p2(x):
    return np.exp(-x)

def p3(x):
    return np.exp(-x/2) / (2 * (1 - np.exp(-1)))

# Inverse CDFs of the importance sampling functions
def inverse_cdf_p1(y):
    return 2 * y

def inverse_cdf_p2(y):
    return -np.log(1 - y)

def inverse_cdf_p3(y):
    return -2 * np.log(1 - y * (1 - np.exp(-1)))

# Monte Carlo integration with importance sampling
N = 10000
uniform_random_numbers = np.random.uniform(0, 1, N)

# p1(x)
x_values = inverse_cdf_p1(uniform_random_numbers)
integral_estimate_p1 = np.mean(f(x_values) / p1(x_values))
variance_p1 = np.var(f(x_values) / p1(x_values))

# p2(x)
x_values = inverse_cdf_p2(uniform_random_numbers)
integral_estimate_p2 = np.mean(f(x_values) / p2(x_values))
variance_p2 = np.var(f(x_values) / p2(x_values))

# p3(x)
x_values = inverse_cdf_p3(uniform_random_numbers)
integral_estimate_p3 = np.mean(f(x_values) / p3(x_values))
variance_p3 = np.var(f(x_values) / p3(x_values))

print(f"Integral estimate with p1(x): {integral_estimate_p1}, Variance: {variance_p1}")
print(f"Integral estimate with p2(x): {integral_estimate_p2}, Variance: {variance_p2}")
print(f"Integral estimate with p3(x): {integral_estimate_p3}, Variance: {variance_p3}")