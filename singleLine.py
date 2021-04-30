# Author: Al Bernstein

# This file uses a rectified linear activation unit - relu function
# from keras to create a line between two points
# to illustrate how the piecewise linear curve fit works

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras import backend as K


fig = plt.figure()
ax1 = fig.add_subplot(111)# initial vector
plt.ylim(-4, 4)
plt.xlim(-5, 5)

plt.title('Line between 0 and 2 implemented with 2 ReLU functions', fontsize = 40)
plt.xlabel('x', fontsize = 40)
plt.ylabel('y(x)', fontsize = 40, rotation = 0, labelpad = 30)
plt.xticks(fontsize = 30, ha = 'left')
plt.yticks(fontsize = 30)


relu_1 = mpatches.Patch(color='black', label='relu(x)')
relu_2 = mpatches.Patch(color='red', label='-relu(x - 2)')
curve = mpatches.Patch(color='green', label='sum')
plt.rcParams["legend.fontsize"] = 30
plt.legend(handles=[relu_1, relu_2, curve])
x1 = np.linspace(-3, 3, 200)

# create a line between x = 0 and x = 2

y1 = K.relu(x1)
y2 = - K.relu(x1-2)


ax1.plot(x1, y1 + y2, linewidth = 5, color = 'green')
ax1.plot(x1, y1, color = 'black', linewidth = 3, linestyle = '--')
ax1.plot(x1, y2, color = 'red',linewidth = 3, linestyle = '--')
plt.show()

