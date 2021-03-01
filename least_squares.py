from linear_systems import *
from numerical_integration import *

# ------------------------------------------------------------
# Approximate function by basis, using least squared rule
# @params:  basis => basis of functions 
#           f =>     the function to approximate
#           a, b =>  interval bounds
# ------------------------------------------------------------
def least_squares(basis, f, a, b):
    x0 = []
    # for orthonormal basis => find coefficients by inner product
    if is_orthonormal_basis(basis, a, b):
        x0 = find_coefficient_orthonormal(basis, f, a, b)
    # for non orthonormal basis => find coefficients linear system
    else:
        x0 = solve_linear_system(basis, f, a, b)

    ff = calculate_approximate(basis, x0) # the approximated function
    error = calculate_error(f, ff, a, b)  # the error value
    return x0, error

# ------------------------------------------------------------
# Check the if a given basis is orthonormal in an interval
# @params:  basis => basis of functions
#           a, b =>  interval bounds
# ------------------------------------------------------------
def is_orthonormal_basis(basis, a, b):
    n = len(basis)
    for i in range(n):
        g = lambda x: basis[i](x) * basis[i](x)
        # calculate inner product to find the basis[i] norma
        normal = simpson_rule(g, a, b)  # <ai, aj>
        if normal != 1:
            return False
        
        for j in range(i, n):
            f = lambda x: basis[i](x) * basis[j](x)
             # calculate inner product
            intgrl = simpson_rule(f, a, b)
            if intgrl != 0:
                return False
    return True

# ------------------------------------------------------------
# Find the coefficient of the approximated function by an 
# orthonormal basis, using numerical integartion
# @params:  basis => basis of functions 
#           f =>     the function to approximate
#           a, b =>  interval bounds
# ------------------------------------------------------------
def find_coefficient_orthonormal(basis, f, a ,b):
    x0 = []
    n = len(basis)
    for i in range(n):
        g = lambda x: basis[i](x) * f(x)
        ci = simpson_rule(g, a, b) # <ai, f>
        x0.append(ci)

    return x0

# ------------------------------------------------------------
# Find the coefficient of the approximated function by non 
# orthonormal basis, using numerical integartion and linear 
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
            ai = simpson_rule(g, a, b) # <ai, aj>
            li.append(ai)
        mat.append(li)
        h = lambda x: basis[i](x) * f(x)
        bi = simpson_rule(h, a, b) # <ai, f>
        vec.append(bi)

    return gauss_seidel(mat, vec)

# ------------------------------------------------------------
# Calculate the approximated function by the basis and the 
# coefficients 
# @params:  basis => basis of functions 
#           x0 =>    coefficients vector
# ------------------------------------------------------------
def calculate_approximate(basis, x0):
    f = lambda x: 0
    for i in range(len(basis)):
        f = lambda x: f(x) + x0[i] * basis[i](x)
    return f

# ------------------------------------------------------------
# Calculate the error: || f - ff || (norma2) 
# @params:  f =>     the original function 
#           ff =>    the approximated function
#           a, b =>  interval bounds
# ------------------------------------------------------------
def calculate_error(f, ff, a, b):
    g = lambda x: f(x) - ff(x)
    gg = lambda x: g(x) * g(x)
    return math.sqrt(simpson_rule(gg, a, b)) # [integral((f-ff)**2)]**0.5