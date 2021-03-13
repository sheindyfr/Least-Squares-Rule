'''
--------------------------------------------------------
Description: running several examples of approximating a
             function in an interval by a function basis
             1) Fourier serias
             2) any basis

Authors: Ayala Barazani & Sheindy Frenkel
Date: 09/03/2021

Files: numerical_integration.py
       linear_systems.py
       least_squares.py
       main.py

Dependencies: numpy, matplotlib
--------------------------------------------------------
'''

from least_squares import *
import matplotlib.pyplot as plt 
import numpy as np 

def create_fourier_basis():
    # Fourier serias
    a1 = lambda x: 1/math.sqrt(2*math.pi)
    a2 = lambda x: (1/math.sqrt(math.pi))*math.cos(x)
    a3 = lambda x: (1/math.sqrt(math.pi))*math.cos(x)
    return [a1, a2, a3]

def create_basis():
    a1 = lambda x: 1
    a2 = lambda x: math.cos(x)
    a3 = lambda x: math.cos(3*x)
    return [a1, a2, a3]

def create_func1():
    f = lambda t: abs(t)
    return f 

def create_func2():
    f = lambda t: t**2 + abs(t)
    return f 

def get_approx_func(basis, coefficient):
    ff = lambda x: calculate_approximate(x, basis, coefficients)
    return ff

def graph(func, x_range, cl='r--'):
    y_range=[]
    for x in x_range:
        y_range.append(func(x))
    plt.plot(x_range, y_range, cl)
    return

def plot_graphs(f, ff):
    rs=1.0
    r=np.linspace(-rs*np.pi,rs*np.pi,80)
    graph(ff,r,cl='r-')
    graph(f,r,cl='b--')
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    basis = create_fourier_basis()
    f = create_func1()
    coefficients, error = least_squares(basis, f, -math.pi, math.pi)
    print(f'\nERROR: {error}')
    ff = get_approx_func(basis, coefficients)
    plot_graphs(f, ff)


    basis = create_basis()
    f = create_func2()
    coefficients, error = least_squares(basis, f, -math.pi, math.pi)
    print(f'\nERROR: {error}')
    ff = get_approx_func(basis, coefficients)
    plot_graphs(f, ff)