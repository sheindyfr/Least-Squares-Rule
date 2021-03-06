import numpy as np
EPSILON = 1e-8

# ------------------------------------------------------------
# Approximate solution of linear system of equation,
# using Gauss-Seidel rule
# @params:  a =>    given coefficient matrix
#           b =>    independent coefficient vector 
#           x0 =>   initial solution vector
#           eps =>  error tolerance
# ------------------------------------------------------------
def gauss_seidel(a, b, x0=None, eps=EPSILON, max_iteration=25):
    print_data(a, b)
    n  = len(a)
    x0 = [0] * n if x0 == None else x0
    x1 = x0[:]

    for __ in range(max_iteration):
        for i in range(n):
            s = sum(-a[i][j] * x1[j] for j in range(n) if i != j) 
            x1[i] = (b[i] + s) / a[i][i]
        if all(abs(x1[i]-x0[i]) < eps for i in range(n)):
            return x1 
        x0 = x1[:]    
    raise ValueError('Solution does not converge')

# ------------------------------------------------------------
# Print the participants of the linear system
# @params:  a =>    given coefficient matrix
#           b =>    independent coefficient vector
# ------------------------------------------------------------
def print_data(a, b):
    print('\n----===[ A ]===----\n')
    for i in a:
        print('\t\t'.join(map(str, i)))
    
    print('\n----===[ b ]===----\n')
    for i in b:
        print(i)

    print('\n----===[ linear system ]===----\n')
    for i in range(len(a)):
        for j in range(len(a[i])):
            print(f'{a[i][j]} * x{j} +', end=" ")
        print(f'= {b[i]}')