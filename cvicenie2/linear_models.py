import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

class LinearModels:

    def loadImages(self, images):
        """Load images"""
        x = []
        for image in images:
            if os.path.exists(image):
                im = np.load(image)#[0,0]['X']
                self.imageSize = np.shape(im)
                x.append(im.flatten())
            else:
                print '%s was not found' % image
        self.nImages = len(x)
        self.X = np.transpose(np.vstack(x))

    def plotImages(self):
        """Plot all images"""
        c = self.imageSize[1]
        for i in range(self.nImages):
            plt.subplot(3,4,i+1)
            plt.imshow(self.X[:,i].reshape(-1,c),cmap=cm.Greys_r)
            #imagesc(eval(mcat([mstring('X'), num2str(i)])), Clim)
            #axis(mstring('off'))
            #title(mcat([mstring('face '), num2str(i)]))i
        plt.show()

    def _computeGI(self):
        """Computing GI weight matrix"""
        #Xt = np.transpose(self.X)
        #Pinv_X = np.dot(np.linalg.inv(np.dot(Xt,self.X)), Xt);
        #self.W_GI = np.dot(self.X,Pinv_X);
        self.W_GI = np.dot(self.X,np.linalg.pinv(self.X));

    def _computeCMM(self):
        self.X_mean = np.transpose(np.mean(np.transpose(self.X)));
        #Xm = X - repmat(X_mean,1,size(X,2));
        Xm = self.X - np.kron(np.ones((1,np.shape(self.X)[1])), self.X_mean)
        self.W_CMM = np.dot(Xm,np.transpose(Xm))/self.nImages;

    def restoreImage(self, imageIndex=None, method='GI', image=None):
        c = self.imageSize[1]
        if image == None:
            if imageIndex == None:
                imageIndex = np.random.randint(0, self.nImages)
        
            corruptedImage = self.X[:,imageIndex].copy()
            corruptedImage = corruptedImage.reshape(-1,1)
            l = np.size(corruptedImage)/c
            corruptedImage[30*l:40*l] = 0
        else:
            corruptedImage = image
        
        if method == 'GI':
            restoredImage,noveltyImage = self.restoreGI(corruptedImage)
        else:
            restoredImage,noveltyImage = self.restoreCMM(corruptedImage)

        plt.subplot(2,2,1)
        plt.imshow(corruptedImage.reshape(-1,c), cmap=cm.Greys_r)
        plt.subplot(2,2,2)
        plt.imshow(restoredImage.reshape(-1,c), cmap=cm.Greys_r)
        plt.subplot(2,2,4)
        plt.imshow(noveltyImage.reshape(-1,c)*-1, cmap=cm.Greys_r)
        plt.show()

    def restoreGI(self, corruptedImage):
        """Restore a corrupted image using General Inverse method"""
        try:
            self.W_GI
        except AttributeError:
            self._computeGI()
        
        restoredImage = np.dot(self.W_GI, corruptedImage)

        noveltyImage = np.dot(np.eye(np.size(corruptedImage))-self.W_GI, corruptedImage);

        return (restoredImage, noveltyImage)

    def restoreCMM(self, corruptedImage):
        """Restore a corrupted image"""
        try:
            self.W_CMM
        except AttributeError:
            self._computeCMM()

        restoredImage = np.dot(self.W_CMM,(corruptedImage - self.X_mean) + self.X_mean)

        noveltyImage = np.ones(np.size(corruptedImage))

        return (restoredImage, noveltyImage)

    def restorePattern(self, method='GI'):
        pattern = np.ones(self.imageSize)
        pattern[10:20,:] = 0
        pattern[40:50,:] = 0
        pattern[:,10:20] = 0
        pattern[:,40:50] = 0
        pattern = pattern.flatten().reshape(-1,1)
        self.restoreImage(pattern, method=method, image=pattern)

def main():
    #images = ['face1.dat', 'face2.dat', 'face3.dat', 'face4.dat', 'face5.dat', 'face6.dat', 'face7.dat', 'face8.dat']
    images = ['img1.dat.npy', 'img2.dat.npy', 'img3.dat.npy', 'img4.dat.npy', 'img5.dat.npy', 'img6.dat.npy', 'img7.dat.npy', 'img8.dat.npy']

    lm = LinearModels()
    lm.loadImages(images)
    lm.plotImages()
    #lm.restoreImage(0, method='CMM')
    #lm.restorePattern(method='CMM')


if __name__ == "__main__":
    main()

