import math

# ------------------------------------------------------------
# Calculate numerical integration for f(x) using trapezoids
# @params:  f =>    the function to be integrated
#           n =>    number of trapezoids   
#           a, b => interval bounds
# ------------------------------------------------------------
def trapezoid_rule(f, a, b, n=100):
    h = (b - a) / float(n)
    integral = 0.5 * h * (f(a) + f(b))

    for i in range(1, int(n)):
        integral = integral + h * f(a + i * h)

    return integral

# ------------------------------------------------------------
# Calculate numerical integration for f(x) using Simpson
# @params:  f =>    the function to be integrated
#           n =>    number of subintervals (must be even)   
#           a, b => interval bounds
# ------------------------------------------------------------
def simpson_rule(f, a, b, n=100):
    if n % 2:
        raise ValueError("n must be even (received n=%d)" % n)

    h = (b - a) / n
    s = f(a) + f(b)

    for i in range(1, n, 2):
        s += 4 * f(a + i * h)
    for i in range(2, n - 1, 2):
        s += 2 * f(a + i * h)

    return s * h / 3


EPSILON = 1e-6
a = -10
b = 10
n = 100
f = lambda x : math.exp(-x**2)

print(trapezoid_rule(f, a, b))
print(simpson_rule(f, a, b))
