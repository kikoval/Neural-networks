
# Code from Chapter 2 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008
# Kristian Valentin, 2011

# Demonstration of the Perceptron on the Pima Indian dataset

from pylab import *
import os
import numpy as np
import pcn

filename = os.path.join(os.getcwd(),'datasets', 'pima-indians-diabetes.data')
pima = np.loadtxt(filename,delimiter=',')

# Plot the first and second values for the two classes
#indices0 = np.where(pima[:,8]==0)
#indices1 = np.where(pima[:,8]==1)
#
#ion()
#plot(pima[indices0,0],pima[indices0,1],'go')
#plot(pima[indices1,0],pima[indices1,1],'rx')

# Perceptron training on the original dataset
print "Output on original data"
p = pcn.pcn(pima[:,:8],pima[:,8:9])
p.pcntrain(0.25,100)
p.confmat(pima[:,:8],pima[:,8:9])

# Various preprocessing steps
pima[np.where(pima[:,0]>8),0] = 8

pima[np.where(pima[:,7]<=30),7] = 1
pima[np.where((pima[:,7]>30) & (pima[:,7]<=40)),7] = 2
pima[np.where((pima[:,7]>40) & (pima[:,7]<=50)),7] = 3
pima[np.where((pima[:,7]>50) & (pima[:,7]<=60)),7] = 4
pima[np.where(pima[:,7]>60)] = 5

pima[:,:8] = pima[:,:8]-pima[:,:8].mean(axis=0)
pima[:,:8] = pima[:,:8]/pima[:,:8].var(axis=0)

#print pima.mean(axis=0)
#print pima.var(axis=0)
#print pima.max(axis=0)
#print pima.min(axis=0)

trainin = pima[::2,:8]
testin = pima[1::2,:8]
traintgt = pima[::2,8:9]
testtgt = pima[1::2,8:9]

# Perceptron training on the preprocessed dataset
print "Output after preprocessing of data"
p1 = pcn.pcn(trainin,traintgt)
p1.pcntrain(0.25,200)
p1.confmat(testin,testtgt)

#show()
