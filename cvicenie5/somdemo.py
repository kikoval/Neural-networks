#!/usr/bin/env python

# Code from Chapter 9 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

# A simple example of using the SOM on a 2D dataset showing the neighbourhood connections

import numpy as np 
import matplotlib.pyplot as plt
import som


# Generating random data from a uniform distribution
nDim = 2
#data = (np.random.rand(2000,nDim)-0.5)*2
data = np.vstack((np.random.normal(0,1,(2000,nDim)),
        np.random.normal(3,1,(1000,nDim))))

# Set up the network and decide on parameters
nNodesEdge = 8
net = som.som(nNodesEdge,nNodesEdge,data,usePCA=1)
step = 0.2

# Plot the data
plt.figure(1)
plt.plot(data[:,0],data[:,1],'.')

# Train the network for 0 iterations (to get the position of the nodes)
net.somtrain(data,0)

# Plot the SOM lattice on top of the input data
for i in range(net.x*net.y):
    neighbours = np.where(net.mapDist[i,:]<=step)
    t = np.zeros((np.shape(neighbours)[1]*2,np.shape(net.weights)[0]))
    t[::2,:] = np.tile(net.weights[:,i],(np.shape(neighbours)[1],1))
    t[1::2,:] = np.transpose(net.weights[:,neighbours[0][:]])
    plt.plot(t[:,0],t[:,1],'g-')
plt.axis('off')

# Repeat the plotting after few iteration of training
plt.figure(2)
plt.plot(data[:,0],data[:,1],'.')

net.somtrain(data,5)

for i in range(net.x*net.y):
    neighbours = np.where(net.mapDist[i,:]<=step)
    t = np.zeros((np.shape(neighbours)[1]*2,np.shape(net.weights)[0]))
    t[::2,:] = np.tile(net.weights[:,i],(np.shape(neighbours)[1],1))
    t[1::2,:] = np.transpose(net.weights[:,neighbours[0][:]])
    plt.plot(t[:,0],t[:,1],'g-')
plt.axis([-1,1,-1,1])
plt.axis('off')

"""
# Repeat the plotting after another few iteration of training
plt.figure(3)
plt.plot(data[:,0],data[:,1],'.')

net.somtrain(data,100)

for i in range(net.x*net.y):
    neighbours = np.where(net.mapDist[i,:]<=step)
    #print neighbours
    #n = tile(net.weights[:,i],(shape(neighbours)[1],1))
    t = np.zeros((np.shape(neighbours)[1]*2,np.shape(net.weights)[0]))
    t[::2,:] = np.tile(net.weights[:,i],(np.shape(neighbours)[1],1))
    t[1::2,:] = np.transpose(net.weights[:,neighbours[0][:]])
    plt.plot(t[:,0],t[:,1],'g-')
    
# Again, hopefully it's the final repetition :) 
#figure(4)
#plot(data[:,0],data[:,1],'.')
#net.somtrain(data,100)
#for i in range(net.x*net.y):
#    neighbours = where(net.mapDist[i,:]<=step)
#    #print neighbours
#    #n = tile(net.weights[:,i],(shape(neighbours)[1],1))
#    t = zeros((shape(neighbours)[1]*2,shape(net.weights)[0]))
#    t[::2,:] = tile(net.weights[:,i],(shape(neighbours)[1],1))
#    t[1::2,:] = transpose(net.weights[:,neighbours[0][:]])
#    plot(t[:,0],t[:,1],'g-')
#    
# No, it's not, this is the final plotting
#figure(5)
#plot(data[:,0],data[:,1],'.')
#net.somtrain(data,100)
#for i in range(net.x*net.y):
#    neighbours = where(net.mapDist[i,:]<=step)
#    #print neighbours
#    #n = tile(net.weights[:,i],(shape(neighbours)[1],1))
#    t = zeros((shape(neighbours)[1]*2,shape(net.weights)[0]))
#    t[::2,:] = tile(net.weights[:,i],(shape(neighbours)[1],1))
#    t[1::2,:] = transpose(net.weights[:,neighbours[0][:]])
#    plot(t[:,0],t[:,1],'g-')
"""
# Show all figures (plots)
plt.show()
