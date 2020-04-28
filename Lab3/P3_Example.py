#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:49:15 2020
Project 3 "help"
@author: rhoover
"""

import skimage #scikit-learn image manipulation #
from skimage import io
from skimage import data
import matplotlib.pyplot as plt #plot utilities #
import numpy as np #numerical computations #
from mpl_toolkits import mplot3d

plt.close('all') #just make sure all plot windows are closed each run #

""" Let's load up some images, reshape them to "vectors", and construct the image data matrix X.
Recall, for this project you do not need to unbias the images, I would recommend just using X for 
everything instead of X_hat """
n = 128
X = np.zeros((128**2,0))
for i in range(n):
    Img = io.imread('P4Images/TrainingImages/Boat64/UnProcessed/img_' + str(i) + '.png', as_gray = True)
    Img = skimage.img_as_float32(Img).reshape(n**2,1)
    #Ivec = Img.reshape(128**2,1)
    X = np.hstack((X,Img))

"""Plot one of the images in X just to be sure things look okay  """
plt.figure(1)
plt.imshow(X[:,50].reshape(n,n), cmap='gray')
plt.axis('off')

"""  Now that we have the Image data matrix (X), we need to compute the eigenvectors of 
the sample covariance matrix (1/n * X * X^T), recall that we can efficiently do this
via the Singular Value Decomposition.  We want to set "full_matrices = False" becasue
we only care about the first 128 coluns of U (the remaining colums simply span the Nullspace of X)"""
U,S,Vt = np.linalg.svd(X,full_matrices=False)

"""  The "eigenimages"" are exactly the columns of U (they just need to be reshaped back to image format) """
fig = plt.figure(2)
fig.suptitle('First 9 Eigenimages for Object 1')
for i in range(8):
    plt.subplot(1,9,i+1)
    plt.axis('off')
    plt.imshow(U[:,i].reshape(n,n),cmap='gray')
plt.pause(0)
    

    
"""  Next let's go ahead and compute the low-dimensional manifold by projecting
the image data back down to a 3-dimensional subspace.

Note 1:  You will likely need
more than 3-dimensions for pose estimation - we just use 3-dimensions so we can 
actually visualize what's going on here. 

Note 2:  Each red dot corresponds to one of the images of object 1 in the low-dimenisonal subspace.
For pose estimation, youd get a new test image, reshape it to a vector, compute p=U[:,0:k].T@Ivec
where in this case, k would be the subspace dimension determined by the energy recovory ratio.
p in this case will be a new k-dimensional point you can compare to each point on the manifold 
(compute the distance between p and all the other points).  The closest point on the manifold to p will
be the estimate of the pose."""
k = 3 #subspace dimension for visualization
M = U[:,0:k].T@X

fig = plt.figure(3)
fig.suptitle('Illustration of pose manifold in $\mathbb{R}^3$')
ax = plt.axes(projection='3d')
ax.plot3D(M[0,:], M[1,:], M[2,:],'blue')
ax.scatter3D(M[0,:], M[1,:], M[2,:],color='red')
ax.set_xlabel('$\phi_1$')
ax.set_ylabel('$\phi_2$')
ax.set_zlabel('$\phi_3$')

