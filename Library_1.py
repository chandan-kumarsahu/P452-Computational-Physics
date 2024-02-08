########################################################################################################################
import math

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
    iterations = 1
    while iterations < max_iter:
        x1 = g(x0)
        if abs(x1 - x0) < tol:
            return x1, iterations
        x0 = x1
        iterations += 1

    raise RuntimeError("Fixed-point iteration did not converge within the maximum number of iterations.")



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

def find_max_abs_f_4th_derivative(f, a, b):
    h = (b - a) / 1000
    x = [a+i*h for i in range(1000)]
    y = []

    for i in range(len(x)):
        # calculate the 4th derivative of f(x) using the central difference method
        y.append(abs((f(x[i] + 2*h) - 4*f(x[i] + h) + 6*f(x[i]) - 4*f(x[i] - h) + f(x[i] - 2*h)) / h**4))
    
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

def calculate_N_s(f, a, b, tol=1e-6):

    fn_s = find_max_abs_f_4th_derivative(f, a, b)

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

def int_simpson(f, a, b, tol=1e-6):

    N=calculate_N_s(f, a, b, tol)
    s=f(a)+f(b)
    h=(b-a)/N
    
    # integration algorithm
    for i in range(1,N):
        if i%2!=0:
            s+=4*f(a+i*h)
        else:
            s+=2*f(a+i*h)
    
    return s*h/3



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
Heat diffusion equation solver using Cranck-Nicolson method.

Parameters:
- T0: Initial temperature profile
- L: Length of the rod
- Tl: Temperature at the left end of the rod
- Tr: Temperature at the right end of the rod
- k: Thermal conductivity
- c: Specific heat
- rho: Density
- t0: Initial time
- tn: Final time
- dt: Time step
- dx: Spatial step
- tol: Tolerance (default = 1e-6)

Returns:
- T: Temperature profile at the final time

"""

def heat_diffusion_CN(T0, L, Tl, Tr, k, c, rho, t0, tn, dt, dx, tol=1e-6):
    N = int(L / dx) + 1
    M = int((tn - t0) / dt) + 1

    T = [[0 for i in range(N)] for j in range(M)]
    T[0] = T0

    alpha = k / (c * rho)
    r = alpha * dt / dx**2

    for j in range(1, M):
        T[j][0] = Tl
        T[j][N-1] = Tr

        A = [[0 for i in range(N)] for j in range(N)]
        B = [0 for i in range(N)]

        for i in range(1, N-1):
            A[i][i-1] = -r
            A[i][i] = 2 + 2*r
            A[i][i+1] = -r
            B[i] = 2*T[j-1][i] + r*(T[j-1][i-1] - 2*T[j-1][i] + T[j-1][i+1])

        A[0][0] = 1
        A[N-1][N-1] = 1

        T[j] = fixed_point_method(lambda x: [B[i] - (A[i][i-1]*x[i-1] + A[i][i]*x[i] + A[i][i+1]*x[i+1]) for i in range(N)], T[j-1], tol=tol)[0]

    return T[M-1]

