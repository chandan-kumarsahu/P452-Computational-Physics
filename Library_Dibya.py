import math

import numpy as np

#********************************************************************************

# Function to round a number to a certain number of decimal places

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


#********************************************************************************

# Fixed-point method

def fixed_point(g, x0, tol=1e-6, max_iter=100):
    iterations = 0
    while True:
        x1 = g(x0)

        # check for convergence
        if abs(x1 - x0) < tol:
            return x1, iterations
        x0 = x1
        iterations += 1
        if iterations > max_iter:
            print("Maximum number of iterations reached. Try a different g(x).")


#********************************************************************************


# Find the maximum value of the absolute value of the 4th derivative of the function.

def find_max_abs_f_4th_derivative(func, a, b, *args):

    # initialize the x and func(x) arrays
    h = (b - a) / 1000
    x = [a+i*h for i in range(1000)]
    f_x = [a+i*h for i in range(1000)]

    # calculate the 4th derivative
    for i in range(len(x)):
        f_x[i] = (abs((func(x[i] + 2*h, *args) - 4*func(x[i] + h, *args) + 6*func(x[i], *args) - 4*func(x[i] - h, *args) + func(x[i] - 2*h, *args)) / h**4))
    
    # return the maximum value
    return max(f_x)


#********************************************************************************


# Calculate the number of subintervals required for the Simpson's rule to achieve a certain error tolerance.

def N_simpson(func, a, b, tol=1e-6, *args):

    # find the maximum value of the 4th derivative
    max_abs_f4 = find_max_abs_f_4th_derivative(func, a, b, *args)

    # Calculation of N from error calculation formula
    N_s=int(((b-a)**5/180/tol*max_abs_f4)**(1/4))
    
    # specila case for N=0 and odd N
    if N_s==0:
        N_s=2    
    if N_s%2!=0:
        N_s+=1

    return N_s


#********************************************************************************

# Numerical integration using the Simpson's rule.

def simpson(func, a, b, tol=1e-8, *args):
    N = N_simpson(func, a, b, tol, *args)
    s = func(a, *args) + func(b, *args)
    h = (b - a) / N
    
    # integration algorithm
    for i in range(1, N):
        if i % 2 != 0:
            s += 4 * func(a + i * h, *args)
        else:
            s += 2 * func(a + i * h, *args)
    
    return s * h / 3


#********************************************************************************


# Function to find the roots of Legendre polynomial and weights for Gaussian Quadrature method.

def gaussian_quadrature_roots_weights(n):
    # Calculate the roots and weights for Gaussian Quadrature using Legendre polynomials
    roots, weights = np.polynomial.legendre.leggauss(n)
    return roots, weights

# Gaussian quadrature for a given order of the Legendre polynomial.

def gauss_quad(func, a, b, ord):

    # Get the roots and weights
    roots, weights = gaussian_quadrature_roots_weights(ord)
    integral = 0

    for i in range(ord):
        # Change of variable from [a, b] to [-1, 1]
        x_i = 0.5 * (b - a) * roots[i] + 0.5 * (a + b)

        # Gaussian quadrature formula
        integral += weights[i] * func(x_i)

    integral *= 0.5 * (b - a)
    return integral

# Gaussian quadrature for a given function and interval.

def Gaussian_quadrature(func, a, b, tol=1e-8):
    ord = 2
    while ord<100:
        # Calculate the integral using Gaussian quadrature for a given order and the next order
        g0 = gauss_quad(func, a, b, ord)
        g1 = gauss_quad(func, a, b, ord+1)

        # Check for convergence
        if abs(g1 - g0) < tol:
            return g1, ord

        ord += 1
    print("Maximum order limit of 100 reached.")


#********************************************************************************

# Runge-Kutta 4th order method for solving an ordinary differential equation.

def RK4_1D_ode(func, y0, x0, xn, h):
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


#********************************************************************************

# Get the matrices A and B for solving the heat diffusion equation using Crank-Nicolson method.

def get_band_matrix(N, sigma):
    A = [[0 for j in range(N)] for i in range(N)]
    B = [[0 for j in range(N)] for i in range(N)]

    # Get the diagonal band matrices A and B
    for i in range(0, N):
        A[i][i] = 2 + 2*sigma
        B[i][i] = 2 - 2*sigma

        # Above and below the diagonal
        if i > 0:
            A[i][i-1] = -sigma
            B[i][i-1] = sigma
        
        if i < N-1:
            A[i][i+1] = -sigma
            B[i][i+1] = sigma

    return A, B


# Solve 1D heat diffusion equation using Crank-Nicolson method.

def crank_nicolson(L, T, dx, dt, Diff, init_cond):

    alpha = Diff * dt / (dx**2)

    # Spatial grid
    x = [i*dx for i in range(int(L/dx)+1)]
    t = [j*dt for j in range(int(T/dt)+1)]

    # Initialize temperature array
    u = [[0 for j in range(int(T/dt)+1)] for i in range(len(x))]

    # Initial condition
    for i in range(len(x)):
        u[i][0] = init_cond(x[i])

    # Get the matrices for solving the matrix using crank-nicolson method
    A, B = get_band_matrix(len(x), alpha)

    u = np.array(u)
    A = np.array(A)
    B = np.array(B)

    for j in range(1, int(T/dt)+1):
        u[:, j] = np.linalg.solve(A, np.dot(B, u[:, j - 1]))

    return u, x, t


#********************************************************************************


# Solve the Poisson equation using Jacobi iterative method for given boundary conditions

def poisson_eqn_solver(Nx, Ny, Lx, Ly, get_BC_poisson):

    Nx = Nx+1
    Ny = Ny+1
    dx = Lx / (Nx)
    dy = Ly / (Ny)

    # Initialize the grid
    x = [i*dx for i in range(Nx)]
    y = [i*dy for i in range(Ny)]

    # Boundary conditions
    u = get_BC_poisson(Nx, Ny, x, y)

    # Source term
    source = [[x[i] * math.exp(y[j]) for j in range(Ny)] for i in range(Nx)]

    # Jacobi method for solving the Poisson equation
    for k in range(1000):
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                u[i][j] = (u[i-1][j] + u[i][j-1] + u[i][j+1] + u[i+1][j] - dx*dy*source[i][j]) /4
    
    return x, y, u


#********************************************************************************
