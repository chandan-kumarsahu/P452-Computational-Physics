def mlcg(seed, a, m, n):
    """Generate n random numbers in range [0,1] using multiplicative linear congruential generator."""
    x = seed
    for _ in range(n):
        x = (a * x) % m
        yield x / m

# Parameters for the MLCG
a1, m1 = 65, 1021
a2, m2 = 572, 16381

# Initial seed
seed = 1

# Generate 10 random numbers using the first set of parameters
random_numbers1 = list(mlcg(seed, a1, m1, 10))
print(f"Random numbers with a={a1}, m={m1}: {random_numbers1}")

# Generate 10 random numbers using the second set of parameters
random_numbers2 = list(mlcg(seed, a2, m2, 10))
print(f"Random numbers with a={a2}, m={m2}: {random_numbers2}")

##plotting the histogram

