########################################################################################################################
import math

import numpy as np

########################################################################################################################

"""
Root finding using fixed-point iteration method.

Parameters:
- g(x): The function for which we want to find the root
- x0: Initial guess for the root
- tol: Tolerance (default = 1e-6)
- max_iter: Maximum number of iterations (default = 100)

Returns:
- root: Approximate root found by the fixed-point iteration.
- iterations: Number of iterations performed.
"""

def fixed_point_method(g, x0, tol=1e-6, max_iter=100):
    for iterations in range(1, max_iter):
        x1 = g(x0)
        if abs(x1 - x0) < tol:
            return x1, iterations
        x0 = x1
    raise RuntimeError("Fixed-point iteration did not converge within the maximum number of iterations. Try a different initial guess of g(x).")



########################################################################################################################


"""
Find the maximum value of the absolute value of the 4th derivative of the function.

Parameters:
- f: The function for which we want to find the maximum value of the absolute value of the 4th derivative
- a: Lower limit of the interval
- b: Upper limit of the interval

Returns:
- Maximum value of the absolute value of the 4th derivative of the function
"""

def find_max_abs_f_4th_derivative(f, a, b, *args):
    h = (b - a) / 1000
    x = [a+i*h for i in range(1000)]
    y = []

    for i in range(len(x)):
        # calculate the 4th derivative of f(x) using the central difference method
        y.append(abs((f(x[i] + 2*h, *args) - 4*f(x[i] + h, *args) + 6*f(x[i], *args) - 4*f(x[i] - h, *args) + f(x[i] - 2*h, *args)) / h**4))
    
    return max(y)



"""
Calculate the number of subintervals required for the Simpson's rule to achieve a certain error tolerance.

Parameters:
- f: The function to be integrated
- a: Lower limit of integration
- b: Upper limit of integration
- eps: Tolerance (default = 1e-6)

Returns:
- N_s: Number of subintervals
"""

def calculate_N_s(f, a, b, tol=1e-6, *args):

    fn_s = find_max_abs_f_4th_derivative(f, a, b, *args)

    # Calculation of N from error calculation formula
    N_s=int(((b-a)**5/180/tol*fn_s)**0.25)
    
    if N_s==0:
        N_s=2
    
    # Special case with simpson's rule
    # It is observed for simpson rule for even N_s, it uses same value
    # but for odd N_s, it should be 1 more else the value is coming wrong
    if N_s%2!=0:
        N_s+=1

    return N_s



'''
Numerical integration using the Simpson's rule.

Parameters:
- f: The function to be integrated
- a: Lower limit of integration
- b: Upper limit of integration
- tol: Tolerance (default = 1e-6)

Returns:
- I: Approximate value of the integral
'''

def int_simpson(f, a, b, tol=1e-8, *args):
    N = calculate_N_s(f, a, b, tol, *args)
    s = f(a, *args) + f(b, *args)
    h = (b - a) / N
    
    # integration algorithm
    for i in range(1, N):
        if i % 2 != 0:
            s += 4 * f(a + i * h, *args)
        else:
            s += 2 * f(a + i * h, *args)
    
    return s * h / 3


########################################################################################################################




########################################################################################################################


"""
Runge-Kutta 4th order method for solving an ordinary differential equation.

Parameters:
- func: Function representing the differential equation dy/dt = func(t, y).
- y0: Initial value of the dependent variable.
- t0: Initial value of the independent variable.
- tn: Final value of the independent variable.
- h: Step size.

Returns:
- x_values: List of time values.
- y_values: List of corresponding dependent variable values.
"""

def ODE_1D_RK4(func, y0, x0, xn, h):
    x = [x0]
    y = [y0]

    while x0 < xn:
        k1 = h * func(x0, y0)
        k2 = h * func(x0 + 0.5 * h, y0 + 0.5 * k1)
        k3 = h * func(x0 + 0.5 * h, y0 + 0.5 * k2)
        k4 = h * func(x0 + h, y0 + k3)

        y0 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        x0 = x0 + h

        x.append(x0)
        y.append(y0)

    return x, y



########################################################################################################################


def get_matrix_heat_diff(N, sigma):
    """
    Get the matrices A and B for solving the heat diffusion equation using

    Parameters:
    - N: Number of spatial grid points
    - sigma: alpha*dt/dx^2

    Returns:
    - A: Matrix A
    - B: Matrix B
    """

    A = [[0 for j in range(N)] for k in range(N)]
    B = [[0 for j in range(N)] for k in range(N)]

    for i in range(0, N):
        A[i][i] = 1 + 2*sigma
        B[i][i] = 1 - 2*sigma
        if i > 0:
            A[i][i-1] = -sigma
            B[i][i-1] = sigma
        if i < N-1:
            A[i][i+1] = -sigma
            B[i][i+1] = sigma

    return A, B



def crank_nicolson_heat_diffusion(L, T, dx, dt, Diff, init_cond):
    """
    Solve 1D heat diffusion equation using Crank-Nicolson method.

    Parameters:
    - L: Length of the rod
    - T: Total time
    - dx: Spatial step size
    - dt: Time step size
    - Diff: Thermal diffusivity

    Returns:
    - u: Temperature distribution over space and time
    - x: Spatial grid
    - t: Time grid
    """

    alpha = Diff * dt / (2 * dx**2)

    # Spatial grid
    x = [i*dx for i in range(int(L/dx)+1)]
    t = [i*dt for i in range(int(T/dt)+1)]

    # Initialize temperature array
    Temp = [[0 for j in range(len(x))] for k in range(int(T/dt)+1)]

    # Initial condition
    for i in range(len(x)):
        Temp[0][i] = init_cond(x[i])

    # Get the matrices for solving the matrix using crank-nicolson method
    A, B = get_matrix_heat_diff(len(x), alpha)

    Temp = np.array(Temp)
    A = np.array(A)
    B = np.array(B)

    for i in range(1, int(T/dt)+1):
        Temp[i, :] = np.linalg.solve(A, np.dot(B, Temp[i - 1, :]))

    return Temp, x, t



########################################################################################################################


