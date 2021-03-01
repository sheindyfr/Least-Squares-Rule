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
        print("orthogonal basis")
        x0 = find_coefficient_orthonormal(basis, f, a, b)
    # for non orthonormal basis => find coefficients linear system
    else:
        print("non-orthogonal basis")
        x0 = solve_linear_system(basis, f, a, b)

    ff = lambda x: calculate_approximate(x, basis, x0) # the approximated function
    error = calculate_error(f, ff, a, b)  # the error value
    return x0, error

# ------------------------------------------------------------
# Check the if a given basis is orthonormal in an interval
# @params:  basis => basis of functions
#           a, b =>  interval bounds
# ------------------------------------------------------------
def is_orthonormal_basis(basis, a, b):
    n = len(basis)
    is_normal_one = True
    is_integral_zero = True
    for i in range(n):
        g = lambda x: basis[i](x) * basis[i](x)
        # calculate inner product to find the basis[i] norma
        normal = inner_product(g, a, b)  # <ai, aj>
        if abs(normal - 1) > EPSILON:
            is_normal_one = False
        
        for j in range(i+1, n):
            f = lambda x: basis[i](x) * basis[j](x)
             # calculate inner product
            intgrl = inner_product(f, a, b)
            print('integral {},{} = {}'.format(i, j, intgrl))
            if abs(intgrl) > EPSILON:
                is_integral_zero = False
    if is_integral_zero and is_normal_one:
        return True
    
# ------------------------------------------------------------
# Find the coefficient of the approximated function by an 
# orthonormal basis, using numerical integration
# @params:  basis => basis of functions 
#           f =>     the function to approximate
#           a, b =>  interval bounds
# ------------------------------------------------------------
def find_coefficient_orthonormal(basis, f, a ,b):
    x0 = []
    n = len(basis)
    for i in range(n):
        g = lambda x: basis[i](x) * f(x)
        ci = inner_product(g, a, b) # <ai, f>
        x0.append(ci)

    return x0

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

    return gauss_seidel(mat, vec)

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
    return math.sqrt(inner_product(gg, a, b)) # [integral((f-ff)**2)]**0.5