# Code from Chapter 2 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

# This is the start of a script for you to complete
import os
import numpy as np
import linreg, pcn

filename = os.path.join(os.getcwd(),'datasets', 'auto-mpg.data')
auto = np.loadtxt(filename,comments='"')

# Normalise the data
auto[:,0:-1] *= 1./auto[:,0:-1].max(axis=0)

# Separate the data into training and testing sets

trainin = auto[0:round(np.shape(auto)[0]*0.4),:]
traintgt = trainin[:,-1].reshape(-1,1)
trainin = trainin[:,0:-1]

testin = auto[round(np.shape(auto)[0]*0.4):,:]
testtgt = testin[:,-1].reshape(-1,1)
testin = testin[:,0:-1]


# This is the training part
beta = linreg.linreg(trainin,traintgt)
testin = np.concatenate((testin,-np.ones((np.shape(testin)[0],1))),axis=1)
testout = np.dot(testin,beta)
error = np.sum((testout - testtgt)**2)
print error

