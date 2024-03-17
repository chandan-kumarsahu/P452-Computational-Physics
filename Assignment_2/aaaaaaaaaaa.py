import math
from time import time


def transpose(A):
    #if a 1D array, convert to a 2D array = matrix
    if not isinstance(A[0],list):
        A = [A]
 
    #Get dimensions
    r = len(A)
    c = len(A[0])

    #AT is zeros matrix with transposed dimensions
    AT = make_matrix(c, r)

    #Copy values from A to it's transpose AT
    for i in range(r):
        for j in range(c):
            AT[j][i] = A[i][j]

    return AT


def scaler_matrix_multiplication(c,A):
    cA = make_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[i])):
            cA[i][j] = c * A[i][j]
    return cA
    

def scaler_matrix_division(c,A):
    cA = make_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[i])):
            cA[i][j] = A[i][j]/c
    return cA


def matrix_multiplication(A, B):
    AB =  [[0.0 for j in range(len(B[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[i])):
            add = 0
            for k in range(len(A[i])):
                multiply = (A[i][k] * B[k][j])
                add = add + multiply
            AB[i][j] = add
    return (AB)





def matrix_addition(A, B):
    
    ra = len(A)
    ca = len(A[0])
    rb = len(B)
    cb = len(B[0])
    
    if ra != rb or ca != cb:
        raise ArithmeticError('Matrices are NOT of the same dimensions!.')
    
    C = make_matrix(ra, cb)
    
    for i in range(ra):
        for j in range(cb):
            C[i][j]=A[i][j] + B[i][j]
    return C

def matrix_substraction(A, B):
    
    ra = len(A)
    ca = len(A[0])
    rb = len(B)
    cb = len(B[0])
    
    if ra != rb or ca != cb:
        raise ArithmeticError('Matrices are NOT of the same dimensions!.')
    
    C = make_matrix(ra, cb)
    
    for i in range(ra):
        for j in range(cb):
            C[i][j]=A[i][j] - B[i][j]
    return C



def matrix_read(M,mode):
    #read the matrix text files
    a = open(M,mode)
    A = []
    #A matrix
    for i in a:
        A.append([float(j) for j in i.split()])
    return (A)


def matrix_copy(A):
    B = make_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[i])):
            B[i][j] = A[i][j]
    return B



def matrix_print(A):
    for i in A:
        for j in i:
            print(j, end='  ')
        print()

def unit_matrix(A):
    B = [[0 for x in range(A)] for y in range(A)]
    for i in range(len(B)):
        for j in range(len(B[i])):
            if i==j:
                B[i][j]=1
    return B


def make_matrix(N, M):
    I = [[0 for x in range(M)] for y in range(N)]
    return I



####################################################################################################################




def matrix_multiplication_on_the_fly(Afn,B):
    n = int(math.sqrt(len(B)))
    #print('B',len(B))
    #print('n',n)
    m = make_matrix(len(B),1)
    for i in range(len(B)):
        for j in range(len(B)):
            m[i][0] = m[i][0] + (Afn(i,j,n) * B[j][0])
    #print('m',m)
    return m



def conjugate_gradient_on_the_fly(Afn, B, eps):
    x0 = []
    a=[1]
    for i in range(len(B)):
        x0.append(a)
    #print('x01',x0)
    '''
    x0=make_matrix(len(B),1)
    for i in range(len(x0)):
        x0[i][0]=1
    print('B',B)
    print("x0",x0) 
    '''
    xk = matrix_copy(x0)
    

    #r0=b-Ax0
    Ax0 = matrix_multiplication_on_the_fly(Afn, x0)
    #print("Ax0",Ax0)
    rk = matrix_substraction(B, Ax0)
    #print("rk",rk)
    i = 0
    dk = matrix_copy(rk)
    #print("dk",dk)
    
    iteration=[]
    residue=[]
    while math.sqrt(inner_product(rk,rk))>=eps and i <= 1000:# and i in range(len(A)):
        adk = matrix_multiplication_on_the_fly(Afn,dk)
        #print("adk=",adk)
        rkrk = inner_product(rk, rk)
        #print("rkrk = ", rkrk)
        alpha = rkrk/inner_product(dk, adk)
        #print("alpha = ",alpha)
        xk = matrix_addition(xk, scaler_matrix_multiplication(alpha, dk))
        #print("xk1=",xk)
        rk = matrix_substraction(rk, scaler_matrix_multiplication(alpha, adk))
        #print("rk1=",rk)
        beta = inner_product(rk, rk)/rkrk
        dk = matrix_addition(rk, scaler_matrix_multiplication(beta, dk))
        
        i = i+1
        #print("norm=",math.sqrt(inner_product(rk,rk)))
        #print("i=",i)
        iteration.append(i)
        residue.append(math.sqrt(inner_product(rk,rk)))
    return xk, iteration, residue

'''
def conju_norm(A):
    sum=0
    for i in range(len(A)):
        sum = sum + abs(A[i][0])
    return sum
'''
def inner_product(A,B):

    AT = transpose(A)

    C = matrix_multiplication(AT, B)

    return C[0][0]






def conjugate_gradient(A, B, x0, eps):
    #r0 = make_matrix(len(A), 1)
    xk = matrix_copy(x0)
    
    #r0=b-Ax0
    Ax0 = matrix_multiplication(A, x0)
    #print("Ax0",Ax0)
    rk = matrix_substraction(B, Ax0)
    #print("rk",rk)
    i = 0
    dk = matrix_copy(rk)
    #print("dk",dk)
    
    iteration=[]
    residue=[]
    while math.sqrt(inner_product(rk,rk))>=eps and i <= 1000:# and i in range(len(A)):
        adk = matrix_multiplication(A,dk)
        #print("adk=",adk)
        rkrk = inner_product(rk, rk)
        #print("rkrk = ", rkrk)
        alpha = rkrk/inner_product(dk, adk)
        #print("alpha = ",alpha)
        xk = matrix_addition(xk, scaler_matrix_multiplication(alpha, dk))
        #print("xk1=",xk)
        rk = matrix_substraction(rk, scaler_matrix_multiplication(alpha, adk))
        #print("rk1=",rk)
        beta = inner_product(rk, rk)/rkrk
        dk = matrix_addition(rk, scaler_matrix_multiplication(beta, dk))
        
        #i = i+1
        #print("norm=",math.sqrt(inner_product(rk,rk)))
        #print("i=",i)
        iteration.append(i)
        residue.append(math.sqrt(inner_product(rk,rk)))
    return xk, iteration, residue






####################################################################################################################





import matplotlib.pyplot as plt


def matrix_function(x, y, n):
    i1 = x%n
    i2 = y%n
    j1 = x//n
    j2 = y//n
    if x == y:
        return -0.96
    if ((i1+1)%n,j1) == (i2,j2):
        return 0.5
    if (i1,(j1+1)%n) == (i2,j2):
        return 0.5
    if ((i1-1)%n,j1) == (i2,j2):
        return 0.5
    if (i1,(j1-1)%n) == (i2,j2):
        return 0.5
    
    return 0


t1 = time()

n = 15
n2 = n**2
A = []
eps = 1e-8


'''
def make_matrix(N, M):
    I = [[0 for x in range(M)] for y in range(N)]
    return I
M = make_matrix(n2,n2)
for i in range(n2):
    for j in range(n2):
        M[i][j] = matrix_function(i,j,n)

print(M)
'''

I=unit_matrix(n2)
for j in range(n2):
    A1 = [[I[i][j]] for i in range(n2)]
    #print("A",A1)
    A1,it,res=conjugate_gradient_on_the_fly(matrix_function, A1, eps)             # sending the function as argument instead of matrix                                                                                 
    print("column=",j)
    for i in range(n2):
        I[i][j] = A1[i][0]

# f = open("q3_invers_of_matrix.csv","w+")
# #saving the inverse of matrix in text file

# for i in range(len(I)):
#     for j in range(len(I[i])):
#         f.write("{:<15}".format(round(I[i][j],6)))
#         f.write('\t')
#     f.write('\n')
# f.close


# residue plot
plt.plot(it,res,label='Conjugate gradient method by generating matrix on the fly')
plt.xlabel('Iterations')
plt.ylabel('Residue')
plt.legend()
plt.savefig('q3_fig.png')


t2 = time()
print("Time taken for the code to run is",t2-t1)

plt.show()

#print("The inverse of the matrix A::")
#ml.matrix_print(I) 
## inverse of matrix is saved on the file 'q3_invers_of_matrix.csv'
