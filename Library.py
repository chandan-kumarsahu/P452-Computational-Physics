########################################################################################################################
import math

import numpy as np

########################################################################################################################



"""
Round a number to a certain number of decimal places.

Parameters:
- n: Number to be rounded
- decimals: Number of decimal places (default = 0)

Returns:
- Rounded number
"""

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def ROUND(n, decimals=10):
    rounded_abs = round_half_up(abs(n), decimals)
    if n>0:
        return rounded_abs
    elif n<0:
        return(-1)*rounded_abs
    else:
        return 0



########################################################################################################################


"""
Root finding using fixed-point method.

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

# Function to find the roots of Legendre polynomial given order using Newton's method. 
# I have manually calculated the roots and weights here. This was just to show the working of the code. 
# However, all these calculations are not required to do iteratively as the root and weight can be calculated once and saved.

"""
Legendre polynomial function.

Parameters:
- x: Initial guess for the root
- n: Order of the Legendre polynomial

Returns:
- P(x): Legendre polynomial at given x and order n
"""

def legendre_polynomial(x, n):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return ((2 * n - 1) * x * legendre_polynomial(x, n - 1) - (n - 1) * legendre_polynomial(x, n - 2)) / n



"""
Function to find the derivative of Legendre polynomial

Parameters:
- x: Initial guess for the root
- n: Order of the Legendre polynomial

Returns:
- P'(x): Derivative of Legendre polynomial at given x and order n
"""

def legendre_derivative(x, n):
    return n * (x * legendre_polynomial(x, n) - legendre_polynomial(x, n - 1)) / (x**2 - 1)



"""
Function to find the roots of Legendre polynomial of order n using Newton's method.

Parameters:
- initial_guess: Initial guess for the root
- n: Order of the Legendre polynomial

Returns:
- x: Roots of the Legendre polynomial
"""

def find_root(initial_guess, n):
    tolerance = 1e-12
    max_iterations = 1000
    x = initial_guess

    for _ in range(max_iterations):
        f_x = legendre_polynomial(x, n)
        f_prime_x = legendre_derivative(x, n)
        x -= f_x / f_prime_x

        if abs(f_x) < tolerance:
            break

    return x



"""
Function to find the roots and weights of the Gaussian quadrature for a given order of the Legendre polynomial.

Parameters:
- n: Order of the Legendre polynomial

Returns:
- roots: Roots of the Legendre polynomial
- weights: Weights of the Legendre polynomial
"""

def get_roots_weights_gaussian(n):
    guess = [np.cos((2 * i + 1) * np.pi / (2 * n)) for i in range(n)]
    roots = [find_root(guess[i], n) for i in range(n)]
    weights = [2 / ((1 - root**2) * legendre_derivative(root, n)**2) for root in roots]

    return roots, weights


# These 3 functions are not required to be calculated everytime while doing actual calculations as 
# the root and weight calculations can be done once and saved in a file. 
########################################################################################################################


"""
Gaussian quadrature for a given order of the Legendre polynomial.

Parameters:
- f: The function to be integrated
- a: Lower limit of integration
- b: Upper limit of integration
- ord: Order of the Legendre polynomial

Returns:
- Gauss_int: Approximate value of the integral
"""

def gauss_quad(f, a, b, ord):
    roots, weights = get_roots_weights_gaussian(ord)
    Gauss_int = 0
    for i in range(ord):
        x_i = 0.5 * (b - a) * roots[i] + 0.5 * (a + b)
        Gauss_int += weights[i] * f(x_i)
    Gauss_int *= 0.5 * (b - a)
   
    return Gauss_int



"""
Gaussian quadrature for a given function and interval.

Parameters:
- f: The function to be integrated
- a: Lower limit of integration
- b: Upper limit of integration
- tol: Tolerance (default = 1e-8)

Returns:
- GQ1: Approximate value of the integral
- ord+1: Order of the Legendre polynomial
"""

def Gaussian_quadrature(f, a, b, tol=1e-8):
    for ord in range(2, 30):
        GQ0 = gauss_quad(f, a, b, ord)
        GQ1 = gauss_quad(f, a, b, ord+1)
        if abs(GQ1 - GQ0) < tol:
            return GQ1, ord

    return ValueError("Integral did not converge within 30 orders of Legendre polynomials.")



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

"""
Get the matrices A and B for solving the heat diffusion equation using

Parameters:
- N: Number of spatial grid points
- sigma: alpha*dt/dx^2

Returns:
- A: Matrix A
- B: Matrix B
"""

def get_matrix_heat_diff(N, sigma):
    A = [[0 for j in range(N)] for k in range(N)]
    B = [[0 for j in range(N)] for k in range(N)]

    for i in range(0, N):
        A[i][i] = 2 + 2*sigma
        B[i][i] = 2 - 2*sigma
        if i > 0:
            A[i][i-1] = -sigma
            B[i][i-1] = sigma
        if i < N-1:
            A[i][i+1] = -sigma
            B[i][i+1] = sigma

    return A, B


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

def crank_nicolson_heat_diffusion(L, T, dx, dt, Diff, init_cond):

    alpha = Diff * dt / (dx**2)

    # Spatial grid
    x = [i*dx for i in range(int(L/dx)+1)]
    t = [j*dt for j in range(int(T/dt)+1)]

    # Initialize temperature array
    Temp = [[0 for j in range(int(T/dt)+1)] for i in range(len(x))]

    # Initial condition
    for i in range(len(x)):
        Temp[i][0] = init_cond(x[i])

    # Get the matrices for solving the matrix using crank-nicolson method
    A, B = get_matrix_heat_diff(len(x), alpha)

    Temp = np.array(Temp)
    A = np.array(A)
    B = np.array(B)

    for j in range(1, int(T/dt)+1):
        Temp[:, j] = np.linalg.solve(A, np.dot(B, Temp[:, j - 1]))

    return Temp, x, t



########################################################################################################################



"""
Solve the Poisson equation using Jacobi iterative method for given boundary conditions

Parameters:
- n_x: Number of grid points in x-direction
- n_y: Number of grid points in y-direction
- x_length: Length of the domain in x-direction
- y_length: Length of the domain in y-direction
- get_BC_poisson: Function to get the boundary conditions

Returns:
- x: Spatial grid in x-direction
- y: Spatial grid in y-direction
- u: Solution of the Poisson equation
"""

def poisson_eqn_solver(n_x, n_y, x_length, y_length, get_BC_poisson):

    n_x += 1
    n_y += 1

    # Discretization
    dx = x_length / (n_x)
    dy = y_length / (n_y)

    # Initialize grid and boundary conditions
    x = [ i*dx for i in range(n_x)]
    y = [ i*dy for i in range(n_y)]

    u = get_BC_poisson(n_x, n_y, x, y)

    # Source term
    f = [[x[i] * math.exp(y[j]) for j in range(n_y)] for i in range(n_x)]

    # Jacobi iterative method
    for _ in range(1000):
        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                u[i][j] = (u[i - 1][j] + u[i][j - 1] + u[i][j + 1] + u[i + 1][j] - dx * dy * f[i][j]) / 4
    
    return x, y, u



########################################################################################################################

