#!/bin/python
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008
# Kristian Valentin, 2011

from pylab import *
import numpy as np

x = np.arange(-3,10,0.05)
y = 2.5 * exp(-(x)**2/9) + 3.2 * exp(-(x-0.5)**2/4)
# Adding noise
y += np.random.normal(0.0, 1.0, len(y))

# Building the interpolation matrix H
H = np.zeros((len(x),2), float)
H[:,0] = exp(-(x)**2/9)
H[:,1] = exp(-(x-0.5)**2/4)

# Computing weight vector
(w, residuals, rank, s) = np.linalg.lstsq(H,y)

print w
# Plotting
plot(x,y,'.')
plot(x,w[0]*H[:,0]+w[1]*H[:,1],'x')
show()
