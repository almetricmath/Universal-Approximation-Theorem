# Author: Al Bernstein

# perfrom neural fit on normalized data and
# then denormalizes it

import numpy as np
import neuroFit as nf

# function to be approximated
func = lambda x: x**2
#func = lambda x: (1/2)*(5*x**3 - 3*x)

# function string to be used in plots
func_str = '$x^2$'
#func_str = '$(1/2)(5x^3 - 3x$)'

fit = nf.neuroFit()

m = 5  # number neurons
n = 200 # number of points

# generate model using m neurons

a = -2
b = 2

x_fit = np.linspace(a, b, m)
y_fit = func(x_fit)


# normalize the model

c = -1
d = 1

xn, yn, mag = fit.normalize(x_fit, y_fit, a, b, c, d)

# # compute weights and biases analytically for relu activation functions in interval [c, d]
 
coeffs = fit.reluFit(m, xn, yn)

# approximate ( reconstruct ) function and approximation at n points in the interval [c, d]

x = np.linspace(c, d, n)
y = func(x)

func_r = fit.reluReconst(coeffs, x)

# compute mean squared error in the interval [c, d]

MSE = (1/n)*np.sum((y - func_r)**2)

# plot results

y_min = min(func_r)
y_max = max(func_r)

fplot = nf.fitPlot((c, d), (y_min - 1, y_max + 1), func_str, m, n, MSE, 30, 40)
fplot.plot(x, y, func_r)


# denormalize data to the interval [a, b]

xd = fit.changeInterval(x, c, d, a, b)
yd = func(xd)
func_r_d = mag*func_r

# compute mean squared error in the interval [a, b] after denormalization

MSE = (1/n)*np.sum((yd - func_r_d)**2)

# plot results

y_min = min(func_r_d)
y_max = max(func_r_d)

fplot = nf.fitPlot((a, b), (y_min - 1, y_max + 1), func_str, m, n, MSE, 30, 40)
fplot.plot(xd, yd, func_r_d)



