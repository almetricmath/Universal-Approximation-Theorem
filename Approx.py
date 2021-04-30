__author__ = 'Al Bernstein'
__license__ = 'MIT License'

import numpy as np
import neuroFit as nf
import sys

# function to be approximated
func = lambda x: x**2
#func = lambda x: (1/2)*(5*x**3 - 3*x)

# function string to be used in plots
func_str = '$x^2$'
#func_str = '(1/2)(5$x^3$ - 3x)'


fit = nf.neuroFit()

m = 3  # number of neurons
n = 200 # number of points

# generate model using m neurons

a = -2
b = 2

x_fit = np.linspace(a, b, m)
y_fit = func(x_fit)

# compute weights and biases analytically for relu activation functions

coeffs = fit.reluFit(m, x_fit, y_fit)
if coeffs == []:
    sys.exit()

# generate function and approximation at n points in the interval [a, b]

x = np.linspace(a, b, n)
y = func(x)


func_r = fit.reluReconst(coeffs, x)

# compute mean squared error in the interval [a, b]
 
MSE = (1/n)*np.sum((y - func_r)**2)

# rescale x to show what happens outside of [a, b]

x1 = np.linspace(2*a, 2*b, 2*n)
y1 = func(x1)
func_r1 = fit.reluReconst(coeffs, x1)

y_min = min(func_r1)
y_max = max(func_r1)

# plot results

fplot = nf.fitPlot((2*a, 2*b), (y_min - 1, y_max + 1), func_str, m, n, MSE, 30, 40)
fplot.plot(x1, y1, func_r1)
