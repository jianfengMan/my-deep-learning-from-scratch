# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.use('TkAgg')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()
