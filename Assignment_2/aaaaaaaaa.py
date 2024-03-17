def is_matrix(var):
    return isinstance(var, list) and isinstance(var[0], list)

def is_function(var):
    return hasattr(var, '__call__')

# Example usage:
variable1 = [[1, 2], [3, 4]]  # A matrix
variable2 = lambda x: x**2  # A function

print("Variable 1 is a matrix:", is_matrix(variable1))
print("Variable 1 is a function:", is_function(variable1))

print("Variable 2 is a matrix:", is_matrix(variable2))
print("Variable 2 is a function:", is_function(variable2))
