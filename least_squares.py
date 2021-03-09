from linear_systems import *
from numerical_integration import *
import scipy.integrate as integrate

# ------------------------------------------------------------
# Approximate function by basis, using least squared rule
# @params:  basis => basis of functions 
#           f =>     the function to approximate
#           a, b =>  interval bounds
# ------------------------------------------------------------
def least_squares(basis, f, a, b):
    x0 = solve_linear_system(basis, f, a, b)
    ff = lambda x: calculate_approximate(x, basis, x0) # the approximated function
    error = calculate_error(f, ff, a, b)  # the error value
    return x0, error

# ------------------------------------------------------------
# Find the coefficient of the approximated function by non 
# orthonormal basis, using numerical integration and linear 
# system of equation solver 
# @params:  basis => basis of functions 
#           f =>     the function to approximate
#           a, b =>  interval bounds
# ------------------------------------------------------------
def solve_linear_system(basis, f, a, b):
    n = len(basis)
    mat = []
    vec = []
    # build the linear system
    for i in range(n):
        li = [] # line i
        for j in range(n):
            g = lambda x: basis[i](x) * basis[j](x)
            ai = inner_product(g, a, b) # <ai, aj>
            li.append(ai)
        mat.append(li)
        h = lambda x: basis[i](x) * f(x)
        bi = inner_product(h, a, b) # <ai, f>
        vec.append(bi)
    x0 = gauss_seidel(mat, vec)
    print('\n----===[ x ]===----\n')
    for i in x0:
        print(i)
    return x0

# ------------------------------------------------------------
# Calculate the approximated function by the basis and the 
# coefficients 
# @params:  basis => basis of functions 
#           x0 =>    coefficients vector
#           t =>     function variable
# ------------------------------------------------------------
def calculate_approximate(t, basis, x0):
    return sum(x0[i] * basis[i](t) for i in range(len(x0)))

# ------------------------------------------------------------
# Calculate the error: || f - ff || (norma2) 
# @params:  f =>     the original function 
#           ff =>    the approximated function
#           a, b =>  interval bounds
# ------------------------------------------------------------
def calculate_error(f, ff, a, b):
    g = lambda x: f(x) - ff(x)
    gg = lambda x: g(x) * g(x)
    return math.sqrt(integrate.quad(gg, a, b)[0]) # [integral((f-ff)**2)]**0.5