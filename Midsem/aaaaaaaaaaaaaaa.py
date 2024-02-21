import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def wave_solver(xa, xb, ta, tb, nx, nt, k, u0, ut0, plot=True):
    # Calculate grid spacings
    dx = (xb - xa) / (nx - 1)
    dt = (tb - ta) / (nt - 1)

    # Initialize grid
    x = np.linspace(xa, xb, nx)
    t = np.linspace(ta, tb, nt)

    # Initialize solution matrix
    U = np.zeros((nt, nx))

    # Set initial conditions
    U[0, :] = u0(x)
    U[1, :] = U[0, :] + dt * ut0(x)

    # Set up tridiagonal matrix for implicit method
    alpha = k * dt**2 / dx**2
    A = np.diag(np.full(nx, 2*(1 + alpha))) - alpha * \
        np.diag(np.ones(nx-1), 1) - alpha * np.diag(np.ones(nx-1), -1)

    # Time-stepping loop
    for n in range(1, nt-1):
        b = U[n, 1:-1]  # Right-hand side vector
        U[n+1, 1:-1] = np.linalg.solve(A, b)

    if plot:
        # Plot the solution
        X, T = np.meshgrid(x, t)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, T, U, cmap=plt.cm.viridis,
                        linewidth=0, antialiased=False)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x, t)')
        ax.set_title('Solution of Wave Equation')
        plt.show()

    return x, t, U


# Example usage:
def u0(x):
    return np.sin(np.pi * x)

def ut0(x):
    return np.zeros_like(x)

xa, xb = 0, 1
ta, tb = 0, 1
nx, nt = 100, 100
k = 1

wave_solver(xa, xb, ta, tb, nx, nt, k, u0, ut0)
