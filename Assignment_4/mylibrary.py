"""
Author: Swaroop Ramakant Avarsekar

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import math


def montecarlo(f, p, a, b, N):
    integral_sum, er = 0, 0
    x = mlcg(N, a, b)  # Generating N random numbers in the range [a, b]
    for i in range(N):
        integral_sum += f(x[i]) / p(x[i]) 
        er += ( ( (f(x[i])**2) / N) - (f(x[i])/N)**2 )
    er = np.sqrt(er)
    integral = integral_sum / N  # Calculating the average
    return integral, er

def mlcg(num, p=0, q=1, seed=1, a=65, m=1021):
    random_numbers = np.zeros(num)
    x = seed
    for _ in range(num):
        x = ((a * x-1) % m)
        u = p + (q-p) * (x/m)
        random_numbers[_] = u
    return random_numbers


def qr_method(A, tol=1e-10):
    diff = 1e5 #np.inf  # Initialize difference
    Q_total = np.eye(A.shape[0])  # Initialize Q_total

    while diff > tol:
        Q, R = gram_schmidt(A)
        A_prev_diag = np.diag(A).copy()  # Store diagonal of A for comparison
        A = R @ Q
        Q_total = Q_total @ Q
        diff = np.sum(np.abs(np.diag(A) - A_prev_diag))

    eigenvalues = np.diag(A)
    return eigenvalues


def gram_schmidt(A):
    Q = np.zeros_like(A)
    for i in range(A.shape[1]):
        v = A[:, i]
        for j in range(i):
            v = v - np.dot(Q[:, j], A[:, i]) * Q[:, j]
        Q[:, i] = v / np.linalg.norm(v)
    R = np.dot(Q.T, A)
    return Q, R

def power_method(A, x=None, epsilon=1e-9):
    n=300
    # If x is not provided, initialize it as a vector of ones with the same shape as A
    if x is None:
        x = np.ones(A.shape[1])
    # Multiply A with x to get the next iteration vector
    x_new = A @ x
    # Remove the factor from the resulting x matrix
    factor = np.linalg.norm(x_new)
    x_new = x_new / factor
    # print(x_new, factor)
    # Check convergence criteria
    if np.linalg.norm(x_new - x) < epsilon:
        return x_new
    # If n is greater than 1, recursively call the power_method function with n-1 and the updated x
    if n > 1:
        return power_method(A, x_new, epsilon)
    else:
        return x_new
    
def calc_eigenval(A):
    x = power_method(A)
    return x @ A @ x / (x @ x)

def sumik(x,k):
        add=0
        for i in range(len(x)):
            add+=(x[i])**k
        return add

def sumxyik(x,y,k):
    add=0
    for i in range(len(x)):
        add+=((x[i])**k)*y[i]
    return add

def polycoeff(x,y,k): #x-data, y-data, k-degree of polynomial
    X,Y=[],[]
    for i in range(k+1):  # Filling X,Y matrix with 0s accordingly
        rowx=[]
        rowy=[0]
        X.append(rowx)
        Y.append(rowy)
        for j in range(k+1):
            rowx.append(0)
    for i in range(len(X)):  # Filling X matrix with required elements
        for j in range(len(X)):
            if i==j:
                X[i][j]=sumik(x,2*i)

            if i!=j and i>j:
                X[i][j]=sumik(x,i+j)
            X[j][i]=X[i][j]
    for i in range(len(Y)): # Filling Y matrix with required elements
        Y[i][0]=sumxyik(x,y,i)
    return X,Y

def conjugate_Gradient_onthefly(matrix_func, b, tol=1e-6, max_iter=500):
    x = np.zeros(len(b))
    r = b - matrix_func(x)
    d = r
    residue = []
    iterations = 0

    while iterations <= max_iter:
        r_old = np.inner(r, r)
        alpha = r_old/np.inner(d, matrix_func(d))
        x += alpha*d
        r -= alpha*matrix_func(d)

        if np.linalg.norm(r) < tol:
            break

        beta = np.inner(r,r)/r_old
        d = r + beta*d
        residue.append(np.linalg.norm(r))
        iterations += 1

    return x, residue


def conj_grad_onthefly_inverse(matrix_func, n, tol=1e-6): # n is order of matrix
    solution, residue = [], []

    for i in range(n):
        b = np.zeros(n)
        b[i] = 1
        inv, res = conjugate_Gradient_onthefly(matrix_func, b, tol)
        solution.append(inv)
        residue.append(res)
    Residue = np.sqrt(np.sum(np.array(residue)**2, axis=0))

    solution = np.array(solution)
    return solution.T, Residue

def conjugate_gradient(a, b, max_iter=10000, tol=1e-9):
    x = np.zeros(len(b))
    r = b - (a @ x)
    d = r
    iterations = 0
    
    while iterations <= max_iter:
        r_old = np.inner(r, r)
        alpha = r_old / np.inner(d, (a @ d))
        x += d*alpha
        r -= (a @ d)*alpha
        
        if r_old < tol:
            return x, iterations + 1
        
        beta = np.inner(r, r) / r_old
        d = r + beta * d
        iterations += 1
    
    return x, iterations + 1
    
def conjugate_gradient_inverse(a):
    a = np.array(a)
    n = len(a)
    A_inv = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        x, _ = conjugate_gradient(a.T, e)
        A_inv[:, i] = x
    return A_inv.T


def multiply_matrix(a,b):
    m = len(a)
    p = len(b[0])
    
    s = np.zeros((m, p))

    # Matrix multiplication
    for i in range(m):
        for j in range(p):
            for k in range(len(b)):
                s[i][j] += a[i][k] * b[k][j]

    return s

def transpose(mat):
    mat1 = [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]
    return mat1

##################################

def bvp(dydx, dzdx, x, Zh, Zl, yo, y1): # Zh, Zl,-- zeta high, low
    n1,t1,i1=coup_ode2(dydx,dzdx,40,Zl,0,0.01,10) #0.01 is time step
    n2,t2,i2=coup_ode2(dydx,dzdx,40,Zh,0,0.01,10)
    
    num1=n1[-1]  # -1 index to access last element as value of the list obtained from the func coup_ode2
    num2=n2[-1]

    Z=Zl+((Zh-Zl)*(y1-num1))/(num2-num1)

    ylis,p1,p2=coup_ode2(dydx,dzdx,40,Z,0,0.01,10)
    y=ylis[-1]
    
    if y-y1<0.001:
        sol,d1z,d2z=coup_ode2(dydx,dzdx,40,Z,0,0.01,x)
        return sol[-1]
    else:
        if Z>Zh:
            return bvp(dydx,dzdx,x,Zh,Z,yo,y1)
        else:
            return bvp(dydx,dzdx,x,Z,Zl,yo,y1)

def coup_ode2(dxdt,dvdt,x,v,t,dt,tup):  # dt is size, tup is upper limit of t
    X = [x]
    V = [v]
    T = [t]
    i = 0
    
    while t < tup:
        
        k1x=dt*dxdt(x,v,t)
        k1v=dt*dvdt(x,v,t)
        
        k2x=dt*dxdt(x+k1x/2,v+k1v/2,t+dt/2)
        k2v=dt*dvdt(x+k1x/2,v+k1v/2,t+dt/2)
        
        k3x=dt*dxdt(x+k2x/2,v+k2v/2,t+dt/2)
        k3v=dt*dvdt(x+k2x/2,v+k2v/2,t+dt/2)
        
        k4x=dt*dxdt(x+k3x,v+k3v,t+dt)
        k4v=dt*dvdt(x+k3x,v+k3v,t+dt)
        
        x+=(k1x+2*k2x+2*k3x+k4x)/6
        v+=(k1v+2*k2v+2*k3v+k4v)/6
        t+=dt
        
        X.append(x)
        V.append(v)
        T.append(t)
        i += 1
        
    return X, T, i

def heat(ut,ux,nx,nt):
    hx = 2/ nx
    ht=4/nt
    alpha=ht/(hx**2)
    Vo,V1,X=[],[],[]

    if alpha>0.5:
        print('choose different nx,nt')
        return None
    
    for i in range(nx+1):
        Vo.append(ux(i*hx))
        V1.append(0)
    
    X=np.linspace(0,2,len(Vo))

    for j in range(nt):
        for i in range(nx+1):
            if i==0:
                V1[i]=(1-2*alpha)*Vo[i]+alpha*Vo[i+1]
            if i==nx:
                V1[i]=(1-2*alpha)*Vo[i] + Vo[i-1]*alpha
            else:
                V1[i] = Vo[i-1]*alpha + Vo[i]*(1-2*alpha) + Vo[i+1]*alpha

        if j == 0 or j== 10 or j== 50 or j== 100 or j== 200 or j== 500 or j== 1000:
            plt.plot(X,Vo,label='t='+str(j*ht))
            
        for i in range(nx+1):
            Vo[i]=V1[i]

        plt.legend()
        plt.xlabel('Length')
        plt.ylabel('Temperature ($^\circ C$)')

    return Vo


def crank_nicolson(V_prev, alpha, n):
    def B_matrix(n): # n is order of matrix
        B = []
        for i in range(n):
            row = []
            for k in range(n):
                row.append(0)
            B.append(row)
    
            for j in range(n):
                if i == j:
                    B[i][j] = 2
                elif i == (j - 1) or i == (j + 1):
                    B[i][j] = -1
        return B

    def I_matrix(n):
        I=[]
        for i in range(n):
            row = []
            for k in range(n):
                row.append(0)
            I.append(row)
            
            for j in range(n):
                if i == j:
                    I[i][j] = 1    
        return I
        
    I = np.array(I_matrix(n)) 
    B = np.array(B_matrix(n))
    first_matrix = linalg.inv(2*I + alpha*B)
    second_matrix = 2*I - alpha*B
    
    return first_matrix@second_matrix@V_prev

def rk4(dydx, xo, yo, h, xup): #xo, yo are Boundary conditions, h is interval and xup is the upper bound for x.
    y=yo
    i=0 #iteration
    while xo < xup:
        k1 = h*dydx(y,xo)
        k2 = h*dydx(y+k1/2,xo+h/2)
        k3 = h*dydx(y+k2/2,xo+h/2)
        k4 = h*dydx(y+k3,xo+h)
        y += (k1+2*(k2+k3)+k4)/6
        xo += h
        i += 1
    return y, xo#,i

def semi_implicit_euler(dxdt, dvdt, xo, vo, dt, steps):
    x = np.zeros(steps) # Initialize arrays for positions and velocities
    v = np.zeros(steps)

    x[0] = xo
    v[0] = vo

    for i in range(1, steps):
        v[i] = v[i-1] + dvdt(x[i-1], v[i-1], i*dt) * dt
        x[i] = x[i-1] + dxdt(x[i-1], v[i], i*dt) * dt
        
    return x, v

def verlet(dvdt, xo, vo, dt, steps):
    # Initialize arrays for positions and velocities
    x = [0] * steps
    v = [0] * steps

    x[0] = xo
    x[1] = x[0] + vo * dt + 0.5 * dvdt(x[0]) * dt**2
    v[0] = vo

    # Perform Verlet integration
    for i in range(2, steps):
        x[i] = 2 * x[i-1] - x[i-2] + dvdt(x[i-1]) * dt**2
        v[i] = v[i-1] + 0.5 * (dvdt(x[i]) + dvdt(x[i-1])) * dt

    return x, v

def velocity_verlet(dvdt, xo, vo, dt, steps):
    x = [0] * steps  # Initialize lists with zeros
    v = [0] * steps
    
    x[0] = xo
    v[0] = vo
    
    for i in range(1, steps):
        
        x[i] = x[i-1] + v[i-1] * dt + 0.5*dvdt(x[i-1]) * dt**2
        v[i] = v[i-1] + 0.5*(dvdt(x[i]) + dvdt(x[i-1]))* dt
        
    return x, v

def leap_frog(dvdt, xo, vo, dt, steps):
    x = [0] * steps  # Initialize lists with zeros
    v = [0] * steps
    x[0] = xo
    v[0] = vo
                   
    for i in range(1, steps):
        v_half = v[i-1] + dvdt(x[i-1]) * (dt/2)
        x[i] = x[i-1] + v_half * dt
        v[i] = v_half + 0.5 * dvdt(x[i]) * dt

    return x, v

#####LINEAR EQUATIONS########

def lu_decomposition(mat, b): # b is rhs of equation
    def decomp(mat):
        for i in range(len(mat)):
            for j in range(len(mat)):
                if i>0 and i<=j: # transforming upper triangular matrix
                    prod=0
                    for k in range(i):
                        prod+=mat[i][k]*mat[k][j]
                    mat[i][j]=mat[i][j]-prod
                if i>j: #transforming lower triangular matrix
                    prod=0
                    for k in range(j):
                        prod+=mat[i][k]*mat[k][j]
                    mat[i][j]=(mat[i][j]-prod)/mat[j][j]
        return mat

    def fsub(mat,b): #forward substition
        y=[0 for i in range(len(mat))]
        y[0]=b[0][0]
        for i in range(len(mat)):
            prod=0
            for j in range(i):
                prod+=mat[i][j]*y[j]
            y[i]=b[i][0]-prod
        return y
    
    def bsub(mat,y): #backward substition
        x=[0 for i in range(len(mat))]
        x[len(mat)-1]=y[len(mat)-1]/mat[len(mat)-1][len(mat)-1]
        for i in range(len(mat)-1,-1,-1): # decrement
            prod=0
            for j in range(i+1,len(mat)):
                prod+=mat[i][j]*x[j]
            x[i]=(y[i]-prod)/mat[i][i]
        return x
    
    u=decomp(mat)
    u2=fsub(mat,b)
    u3=bsub(mat,u2)
    
    return u3

def cholesky(mat):
    if mat==transpose(mat):
        for i in range(len(mat)):
            add=0
            #print(add)
            for j in range(i):
                add+=(mat[i][j])**2
            mat[i][i]=math.sqrt(mat[i][i]-add)

            for j in range(i+1,len(mat)):
                tot=0
                for k in range(i):
                    tot+=mat[i][k]*mat[k][j]
                mat[j][i]=(mat[j][i]-tot)/mat[i][i]
                mat[i][j] = mat[j][i]

        for i in range(len(mat)): # triangular part zero
            for j in range(len(mat[0])):
                if j>i:
                    mat[i][j]=0
        return mat
    
    else:
        print('Asymmetric matrix, cannot be Cholesky decomposed')

def chol_fsub(mat,coeff):
    y=[]
    for i in range(0,len(mat)):#i=1
        prod=0
        for j in range(i):#for j in range(0,0)
            prod+=(mat[i][j]*y[j])
        y.append((coeff[i][0]-prod)/mat[i][i])
    return y

def chol_bsub(mat1,y):
    x=[0]*len(mat1)
    #print(x)
    for i in range(len(mat1)-1,-1,-1):
        prod=0
        for j in range(i+1,len(mat1)):
            prod+=mat1[i][j]*x[j]
        #print(i)
        x[i]=((y[i]-prod)/mat1[i][i])
    return x

def transpose(mat):
    mat1 = [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]
    return mat1

def print_matrix(m):
    for i in range(len(m)):
        print(m[i])

#***gauss siedel ****#
def ddom(a,b):
    sumlist=[]
    for row in range (0,len(a)):
        add=0
        for col in range (0, len(a[row])):
            add+=abs(a[row][col])
        sumlist.append(add)
    check=0
    for i in range (0,len(a)):
        if abs(a[i][i])>= (sumlist[i]/2): 
            pass
        else:
            for j in range (i+1,len(a)):
                if abs(a[j][i]) >= (sumlist[j]/2): 
                    a[j],a[i]=a[i],a[j] 
                    b[j],b[i]=b[i],b[j]
                    check = 1
    if check==1:  # if check is 1 implies diagonally dominant
        return a,b
    else:
        return print("diagonal dominant not possible")
    
def ddom_already(a,b):
    sumlist=[]
    for row in range (0,len(a)):
        add=0
        for col in range (0, len(a[row])):
            add+=abs(a[row][col])
        sumlist.append(add)
    check=0
    for i in range (0,len(a)):
        if abs(a[i][i])>= sumlist[i]: #already diadonally dominant
            check=1
        else:
            check=0
    return check

def gauss_seidel(c,d):
    a,b=c,d # a,b=ddom(c,d) Consider the second statement if it is not diagonally dominant
    guess=[] # Let this be the guess solution
    prev=[] # let prev list be filled with zeros
    for row in range(len(a)):
        guess.append([0])
        prev.append([0])
        
    count=0
    new=guess
    for somenum in range(100): # setting up iteration limit          
        for i in range(len(a)):
            prev[i][0]=new[i][0] # assigning prev to new
    
        for i in range(len(a)): # Formula 
            temp=b[i][0]
            for j in range(i):
                temp=temp-a[i][j]*new[j][0]  
            for j in range(i+1,len(a)):
                temp=temp-a[i][j]*prev[j][0]
            new[i][0]=temp/a[i][i]
        count+=1
        
        d=0 #precision check
        for num in range(len(prev)):
            d+=abs(new[num][0]-prev[num][0])
        if d<1e-6:
            break
    return count, new

def gauss_jacobi(A, B, X, precision):
    n = len(A)  
    temp = 1  # temporary variable to enter the loop

    while temp > precision * n:
        diff = [0] * n  # store the differences b/w old and new solution
        for i in range(n):
            term = sum(A[i][j] * X[j][0] for j in range(n) if i != j) #sum of terms excluding the diagonal element
            X_new = (1 / A[i][i]) * (B[i][0] - term) #new value for the solution
            diff[i] = abs(X_new - X[i][0])
            X[i][0] = X_new
        temp = sum(diff)

    return X
    
def augment_matrix(a, b): #[a|b]
    
    for i in range(len(a)):
        for j in range(len(b[i])):
            a[i].append(b[i][j])
            
    return a

def gauss_jordan(A, n):
    if n >= len(A):
        return [row[-1] for row in A]

    # Find the largest pivot element to improve numerical stability
    largest_row = max(range(n, len(A)), key=lambda i: abs(A[i][n]))
    A[n], A[largest_row] = A[largest_row], A[n]

    pivot_element = A[n][n]

    # Normalize the pivot row
    for j in range(len(A[n])):
        A[n][j] /= pivot_element

    # Eliminate other rows
    for i in range(len(A)):
        if i != n:
            factor = A[i][n] / A[n][n]
            for j in range(n, len(A[i])):
                A[i][j] -= factor * A[n][j]

    # Move to the next pivot row and return solutions
    return gauss_jordan(A, n + 1)


############ROOT FINDING###################

def bracketing(f,a,b): #a<b
    #global count
    count=0
    if f(a) * f(b) < 0:
        #print(a, b)
        return a,b#,count
    
    if f(a) * f(b) > 0:  #f(a) and f(b) have same 
        
        if abs(f(a)) < abs(f(b)):
            a = a - 0.5 * (b-a)   # shifting a to left
            count += 1
            return bracketing(f,a,b)
        if abs(f(a)) > abs(f(b)): 
            b = b + 0.5 * (b-a)  # shifting b to right
            count += 1
            return bracketing(f,a,b)
        
    if count>11:
        print('choose different intervals')

def bisection(f, a, b, E, D, ctr=0):
    
    a, b = bracketing(f, a, b) #add count if you wish
    #print(round(a,7),round(b,7),ctr)
    if abs(b-a) < E and f(a) < D : #precision check
        return round(a,7)#, ctr#,round(b,7)#,ctr
    
    else:    
        c = (a+b) / 2 # bisecting the interval
        
        if f(c) * f(a) < 0:
            ctr += 1
            return bisection(f, a, c, E, D, ctr) # shifting b
        
        if f(c) * f(b) < 0:
            ctr += 1
            return bisection(f, c, b, E, D, ctr) #shifting a

def regula_falsi(f, a, b, E, D): # E is epsilon, D is delta; precision- tolerance
    c, ctr = b - ( (b-a) * f(b) ) / ( f(b) - f(a) ), 0
    #print(round(c,7),ctr)
    if f(a) * f(b) < 0:
        if abs(b-a) > E or ( f(a) > D and f(b) > D ):  #precision check
            while abs(a-b) > E:
                if f(a) * f(c) < 0:
                    b = c
                if f(b) * f(c) < 0:
                    a = c
                if abs(f(c)) < D:
                    return round(b,6), ctr#, round(c,9),       #upto 6 decimal places      
                c = b - ( (b-a) * f(b) ) / ( f(b) - f(a) )
                ctr += 1    
                print(round(b,7),round(c,7),ctr)
    
    else:   # if the given bracket doesn't satisfy
        print(round(b,7),round(c,7),ctr) 
        a, b = bracketing(f, a, b) #count
        return regula_falsi(f, a, b, E, D)

# def secant_method_test(f, xo, x1): #xo and x1 are two gues vals, 
#     ctr = 0
#     while abs(xo-x1) < 1000: #E = 1000
#         x2 = x1 - (x1-xo) * f(x1) / (f(x1) - f(xo))
#         xo = x1
#         x1 = x2
#         ctr += 1
#         if xo == x1:
#             break 
#     return round(xo,7)#, ctr

def secant_method(f, a, b, E, D):
    ctr = 0
    if f(a) * f(b) < 0:
        while abs(b - a) > 1e-4: # tolerance = 1e-6
            c = a - f(a) * (b - a) / (f(b) - f(a))
            ctr += 1
            if abs(f(c)-f(a))<1e-9: #precision
                return c
            a = b
            b = c
        return round(c, 7)#,ctr
    else:
        print('Change the intervals')
        return None

def newton_raphson(f, fd, x, E, D): #x-guess,E-epsilon,D-delta
    iter = 0 #iteration
    a, b = x - f(x)/fd(x), x
    while abs(a-b) > E and abs(f(b)) > D: #precision check
        b = a
        a = b - f(b)/fd(b)
        iter += 1
        print(round(a,7), round(b,7))
    return round(a,6), iter#, round(b,5)


def one_var_fixedpt(f, x_guess, E): # x_guess is guess value, f is function, E is tolerance
    x = x_guess
    iter = 0 #iteration count
    #print('Iter \t Root')
    while True:
        g = f(x)
        if abs(x-g) < E:
            #print("\n Converged Root:", x)
            return round(x,7)
        x = g
        iter += 1
        #print(iter,'\t', g)

###############INTEGRATION#################

def int_midpt(f, a, b, N):  #a,b are limits; f is integrand; N is no. of iterations
    x = []
    h = (b-a) / N   #width
    area=0
    for i in range(1, N+1):
        x.append( (2 * a + (2 * i - 1) * h ) / 2)
        area += h * f( x[i-1] )  # adding all area
    return area #integral

def int_trapezoid(f, a, b, N):  #a,b are limits; f is integrand; N is no. of iterations
    x = [a]
    h = (b - a) / N  #width
    area = 0
    for i in range(1, N+1):
        x.append(a + i * h)
        area += ( f(x[i]) + f(x[i-1]) )  #sum of parallel sides
    area = h * area / 2   #integral (area of trapezoid)
    return area 

def simpson(f, a, b, N):  #a,b are limits; f is integrand, N is no. of iterations
    h = (b - a) / N   #width
    area = f(a) + f(b)  # first and last lim are applied
    x = [a] # a is just first element to satisfy loop
    for i in range(1,N): # variables ranging in b/w limits
        x.append(a + i * h)
        if i%2 == 0:
            area += (2 * f(x[i]))  
        else:
            area += (4 * f(x[i]))    
    area = h * area / 3
    return area #integral

        
        