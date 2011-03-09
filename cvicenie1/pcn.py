
# Code from Chapter 2 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008
# Kristian Valentin <valentin@fmph.uniba.sk>, 2011

import numpy as np

class pcn:
    """ A basic Perceptron"""
    
    def __init__(self,inputs,targets,activationFunction='linear'):
        """ Constructor """
        # Set up network size
        if np.ndim(inputs)>1:
            self.nIn = np.shape(inputs)[1]
        else: 
            self.nIn = 1
    
        if np.ndim(targets)>1:
            self.nOut = np.shape(targets)[1]
        else:
            self.nOut = 1

        self.inputs = inputs
        self.targets = targets

        self.nData = np.shape(inputs)[0]
    
        # Initialise network
        self.weights = np.random.rand(self.nIn+1,self.nOut)*0.1-0.05

        # Activation function and its derivative
        functions = {'linear': lambda x: np.where(x>0,1.,0.),
                     'sigmoid': lambda x: 1./(1.+np.exp(-x))}
        functionsDeriv = {'linear': lambda x: 1.,
                          'sigmoid': lambda x: x*(1.-x)}
        if activationFunction in functions:
            self.f = functions[activationFunction]
            self.fd = functionsDeriv[activationFunction]
        else:
            self.f = functions['linear']
            self.fd = functionsDeriv['linear']

    def _computeError(self, inputs, targets):
        """Compute least square error"""
        return .5*np.sum((targets-self._pcnfwd(inputs))**2)

    def onlineTrain(self, eta, verbose=False):
        """Online training"""
        inputs = np.concatenate((self.inputs,-np.ones((self.nData,1))),axis=1)
        targets = self.targets
        
        change = range(self.nData)
        error = self._computeError(inputs, targets)
        olderr = np.inf

        # Stopping criterion can vary
        # error > 0.1
        # classiffError > 0
        while np.abs(olderr-error)/error > 0.01:
            if verbose:
                print "Error: %.4f" % error
            for i in range(np.shape(inputs)[0]):
                for j in range(np.shape(self.weights)[0]):
                    output = self._pcnfwd(inputs[i,:])
                    self.weights[j] += eta*(targets[i]-output)*inputs[i,j]*self.fd(output)
            
            olderr = error
            error = self._computeError(inputs, targets)

            # Randomise order of inputs
            np.random.shuffle(change)
            inputs = inputs[change,:]
            targets = targets[change,:]

    def pcntrain(self,eta,nIterations,verbose=False):
        """ Online learning using matrix notation"""    
        # Add the inputs that match the bias node
        inputs = np.concatenate((self.inputs,-np.ones((self.nData,1))),axis=1)
        targets = self.targets
        # Training
        change = range(self.nData)

        for n in range(nIterations):
            
            self.outputs = self._pcnfwd(inputs)
            self.weights += eta*np.dot(np.transpose(inputs),self.fd(self.outputs)*(targets-self.outputs))
            if verbose:
                print "Iteration: ", n
		print self.weights
			
		activations = self._pcnfwd(inputs)
		print "Final outputs are:"
		print activations
        
            # Randomise order of inputs
            np.random.shuffle(change)
            inputs = inputs[change,:]
            targets = targets[change,:]

    def _pcnfwd(self,inputs):
        """ Run the network forward """

        outputs = np.dot(inputs,self.weights)

        # Threshold the outputs
        return self.f(outputs)

    def output(self, inputs):
        """ Run the trained network """

        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self._pcnfwd(inputs)

        # Threshold the outputs
        return self.f(outputs)

    def confmat(self,inputs,targets):
        """Confusion matrix"""

        nClasses = np.shape(targets)[1]

        if nClasses==1:
            nClasses = 2
            outputs = self.output(inputs)
        else:
            # Add the inputs that match the bias node
            inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)
        
            outputs = np.dot(inputs,self.weights)
        
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        outputs = np.where(outputs>0.6,1,0)

        cm = np.zeros((nClasses,nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print cm
        print "Classification accuracy: %.2f" % (np.trace(cm)/np.sum(cm)*100.)
        
def logic():
    """ Demo"""
    """ Run AND, OR and XOR logic functions"""
    a = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]],dtype=float)
    b = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,1]],dtype=float)
    c = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]],dtype=float)
    
    q = pcn(a[:,0:2],a[:,2:],'sigmoid')
    q.onlineTrain(0.25, True)
    #q.pcntrain(0.25,30)
    q.confmat(a[:,0:2],a[:,2:])
    
    q = pcn(b[:,0:2],b[:,2:])
    q.onlineTrain(0.25, True)
    #q.pcntrain(0.25,20)
    q.confmat(b[:,0:2],b[:,2:])

    p = pcn(c[:,0:2],c[:,2:],'sigmoid')
    q.onlineTrain(0.25, True)
    #p.pcntrain(0.05,100)
    p.confmat(c[:,0:2],c[:,2:])
    
if __name__ == "__main__":
    logic()
