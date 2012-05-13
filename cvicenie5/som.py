#!/usr/bin/env python

# Code from Chapter 9 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008
# Kristian Valentin, 2011

import numpy as np
import pca

class som:
    """A Basic 2D Self-Organising Map
    The map connections can be initialised randomly or with PCA"""
    def __init__(self,x,y,inputs,eta_b=0.3,eta_n=0.1,nSize=0.5,alpha=1,usePCA=1,useBCs=0,eta_bfinal=0.03,eta_nfinal=0.01,nSizefinal=0.05):
        self.nData = np.shape(inputs)[0]
        self.nDim = np.shape(inputs)[1]
        
        # output map size
        # TODO make more universal
        self.mapDim = 2
        self.x = x
        self.y = y

        self.eta_b = eta_b
        self.eta_bfinal = eta_bfinal
        self.eta_n = eta_n
        self.eta_nfinal = eta_nfinal
        self.nSize = nSize
        self.nSizefinal = nSizefinal
        self.alpha = alpha

        self.map = np.mgrid[0:1:complex(0,x),0:1:complex(0,y)]
        self.mapDim = 2
        self.map = np.reshape(self.map,(2,x*y))
        
        # weights initialization
        if usePCA:
            dummy1,dummy2,evals,evecs = pca.pca(inputs,2)
            self.weights = np.zeros((self.nDim,x*y))
            for i in xrange(self.x*self.y):
                for j in range(self.mapDim):
                    self.weights[:,i] += (self.map[j,i]-0.5)*2*evecs[:,j]            
        else:
            # random values from the interval <-1,1>
            self.weights = (np.random.rand(self.nDim,x*y)-0.5)*2    
        
        # pre-computing the map distances
        self.mapDist = np.zeros((self.x*self.y,self.x*self.y))
        if useBCs:
            for i in xrange(self.x*self.y):
                for j in xrange(i+1,self.x*self.y):
                    xdist = np.min([(self.map[0,i]-self.map[0,j])**2,(self.map[0,i]+1+1./self.x-self.map[0,j])**2,(self.map[0,i]-1-1./self.x-self.map[0,j])**2,(self.map[0,i]-self.map[0,j]+1+1./self.x)**2,(self.map[0,i]-self.map[0,j]-1-1./self.x)**2])
                    ydist = np.min([(self.map[1,i]-self.map[1,j])**2,(self.map[1,i]+1+1./self.y-self.map[1,j])**2,(self.map[1,i]-1-1./self.y-self.map[1,j])**2,(self.map[1,i]-self.map[1,j]+1+1./self.y)**2,(self.map[1,i]-self.map[1,j]-1-1./self.y)**2])
                    self.mapDist[i,j] = np.sqrt(xdist+ydist)
                    self.mapDist[j,i] = self.mapDist[i,j]                
        else:
            for i in xrange(self.x*self.y):
                for j in xrange(i+1,self.x*self.y):
                    self.mapDist[i,j] = np.sqrt((self.map[0,i] - self.map[0,j])**2 + (self.map[1,i] - self.map[1,j])**2)
                    self.mapDist[j,i] = self.mapDist[i,j]
                
    def somtrain(self,inputs,nIterations):
        """Training"""
        self.eta_binit = self.eta_b
        self.eta_ninit = self.eta_n
        self.nSizeinit = self.nSize

        for iterations in range(nIterations):
            for i in range(self.nData):
                best,activation = self.somfwd(inputs[i,:])
                # Update the weights of the best match
                self.weights[:,best] += self.eta_b * (inputs[i,:] - self.weights[:,best])
                # Find the neighbours and update their weights
                neighbours = np.where(self.mapDist[best,:]<=self.nSize,1,0)
                neighbours[best] = 0
                self.weights += self.eta_n * neighbours*np.transpose((inputs[i,:] - np.transpose(self.weights)))
            # Modify learning rates
            self.eta_b = self.eta_binit*np.power(self.eta_bfinal/self.eta_binit,float(iterations)/nIterations)
            self.eta_n = self.eta_ninit*np.power(self.eta_nfinal/self.eta_ninit,float(iterations)/nIterations)
        
            # Modify neighbourhood size
            self.nSize = self.nSizeinit*np.power(self.nSizefinal/self.nSizeinit,float(iterations)/nIterations)
    
    def somfwd(self, inputs):
        """Euclidean distance metric"""
        # compute vector of activations for each output neuron
        activations = np.sum((np.transpose(np.tile(inputs,(self.x*self.y,1)))-self.weights)**2,axis=0)
        # get the index of the winner
        best = np.argmin(activations)
        return best,activations
