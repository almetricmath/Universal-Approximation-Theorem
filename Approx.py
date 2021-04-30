# Author: Al Bernstein

import numpy as np
import neuroFit as nf

# function to be approximated
func = lambda x: x**2
#func = lambda x: (1/2)*(5*x**3 - 3*x)

# function string to be used in plots
func_str = '$x^2$'
#func_str = '(1/2)(5$x^3$ - 3x)'


fit = nf.neuroFit()

m = 20
n = 200

# generate function

x_fit = np.linspace(-1, 1, m)
y_fit = func(x_fit)

# compute weights analytically

coeffs = fit.reluFit(m, x_fit, y_fit)

# generate function and approximation at n points

x = np.linspace(-1, 1, n)
y = func(x)
func_r = fit.reluReconst(coeffs, x)

# compute mean squared error in interval
 
MSE = (1/n)*np.sum((y - func_r)**2)

# rescale x to show what happens outside of [-1, 1]


x1 = np.linspace(-2, 2, 2*n)
y1 = func(x1)
func_r1 = fit.reluReconst(coeffs, x1)


# plot results

fplot = nf.fitPlot((-3, 3), (-2, 2), func_str, m, n, MSE, 30, 40)
fplot.plot(x1, y1, func_r1)



