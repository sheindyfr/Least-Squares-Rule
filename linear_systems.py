def gauss_seidel(a, b, x0=None, eps=1e-5, max_iteration=25):
    """
    m  : list of list of floats : coefficient matrix
    x0 : list of floats : initial guess
    eps: float : error tolerance
    max_iteration: int
    """
    n  = len(a)
    print(n)
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

print(gauss_seidel(a, b))
