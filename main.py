from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import hashlib
import pandas as pd
from classifiers.linear_classifier import LinearSVM
from six.moves.urllib.request import urlretrieve
from data_utils import load_CIFAR10
from data_utils import load_mydata
from classifiers.linear_svm import svm_loss_naive
from gradient_check import grad_check_sparse
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import KernelPCA
input_width=32
input_height=32

X_train, y_train =load_mydata("C:/Users/user/Desktop/train.txt",input_width,input_height,profect=True)
X_test, y_test =load_mydata("C:/Users/user/Desktop/test.txt",input_width,input_height,profect=True)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
#print('Test data shape: ', X_test.shape)
#print('Test labels shape: ', y_test.shape)

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
def gen_class(number_class):
  r=range(number_class)
  classes=list(r)
  return classes  
class_num=50
classes=gen_class(class_num)
samples_per_class = 10

def visualize_data(dataset, classes, samples_per_class):
    num_classes = len(classes)
    for y, cls in enumerate(classes):
      idxs = np.random.choice(len(y_train), samples_per_class, replace=False)
      for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(dataset[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
          plt.title(cls)
    plt.show()



#visualize_data(X_train, classes, samples_per_class)

# Split the data into train, val, and test sets. In addition we will
# create a small development set as a subset of the training data;
# we can use this for development so our code runs faster.
num_training = 40000
num_validation = 1000
num_test = 300
num_dev = 150

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We will also make a development set, which is a small subset of
# the training set.
mask = np.random.choice(num_training, num_dev, replace=True)
X_dev = X_train[mask]
y_dev = y_train[mask]
print(y_dev)

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train.transpose(0,3,1,2), (X_train.shape[0], -1))
X_val = np.reshape(X_val.transpose(0,3,1,2), (X_val.shape[0], -1))
X_test = np.reshape(X_test.transpose(0,3,1,2), (X_test.shape[0], -1))
X_dev = np.reshape(X_dev.transpose(0,3,1,2), (X_dev.shape[0], -1))
#transformer = KernelPCA(n_components=10, kernel='linear')
#X_train = transformer.fit_transform(X_train)
#X_val= transformer.fit_transform(X_val)
#X_test= transformer.fit_transform(X_test)
#X_dev= transformer.fit_transform(X_dev)


# As a sanity check, print out the shapes of the data
print('Training data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)
print('dev data shape: ', X_dev.shape)

# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
print(mean_image)
mean_image.astype('uint8')
#plt.figure(figsize=(4,4))
#plt.imshow(mean_image.reshape((input_width,input_height,-1)).astype('uint8')) # visualize the mean image
#plt.show()

# second: subtract the mean image from train and test data
#X_train -= mean_image
#X_val -= mean_image
#X_test -= mean_image
#X_dev -= mean_image

# Visualize some samples of each category after preprocessing
#visualize_data(X_train, classes, samples_per_class)

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train =np.array(np.hstack([X_train, np.ones((X_train.shape[0], 1))])) 
X_val = np.array(np.hstack([X_val, np.ones((X_val.shape[0], 1))]))
X_test = np.array(np.hstack([X_test, np.ones((X_test.shape[0], 1))]))
X_dev = np.array(np.hstack([X_dev, np.ones((X_dev.shape[0], 1))]))
print("type is:",type(X_train),type(X_val),type(X_test),type(X_dev))

print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)

# Generate a random SVM weight matrix of small numbers
W = np.random.randn(X_train.shape[1],100) * 0.00001 

# For debugging purpose we can calculate the loss with very low W and no regularization
# The result should be near 9 (#number_class - 1)

loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.)
print('loss: %f' % (loss, ))

# Compute the loss and its gradient at W.
loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)

# Numerically compute the gradient along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should match
# almost exactly along all dimensions.
from gradient_check import grad_check_sparse
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical,grad_numerical_list,grad_analytic_LIST,rel_error_list = grad_check_sparse(f, W, grad)

# do the gradient check once again with regularization turned on
# you didn't forget the regularization gradient did you?
loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad)

# Next implement the function svm_loss_vectorized; for now only compute the loss;
# we will implement the gradient in a moment.
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))

from classifiers.linear_svm import svm_loss_vectorized
tic = time.time()
loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# The losses should match but your vectorized implementation should be much faster.
print('difference: %f' % (loss_naive - loss_vectorized))


# Complete the implementation of svm_loss_vectorized, and compute the gradient
# of the loss function in a vectorized way.

# The naive implementation and the vectorized implementation should match, but
# the vectorized version should still be much faster.
tic = time.time()
_, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Naive loss and gradient: computed in %fs' % (toc - tic))

tic = time.time()
_, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('Vectorized loss and gradient: computed in %fs' % (toc - tic))

# The loss is a single number, so it is easy to compare the values computed
# by the two implementations. The gradient on the other hand is a matrix, so
# we use the Frobenius norm to compare them.
difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('difference: %f' % difference)
# In the file linear_classifier.py, implement SGD in the function
# LinearClassifier.train() and then run it with the code below.

from classifiers.linear_classifier import LinearSVM
test_accuracy_list=[]
train_accuracy_list=[]
val_accuracy_LIST=[]



svm = LinearSVM()
tic = time.time()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                      num_iters=1500, verbose=True, X_val=X_val, y_val=y_val)
toc = time.time()
print('That took %fs' % (toc - tic))

# A useful debugging strategy is to plot the loss as a function of
# iteration number:
plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
y_train_pred,score_train = svm.predict(X_train)
train_accuracy=np.mean(y_train == y_train_pred)
print('training accuracy: %f' % (train_accuracy, ))
y_val_pred,score_val = svm.predict(X_val)
val_accuracy=np.mean(y_val == y_val_pred)
print('validation accuracy: %f' % (val_accuracy, ))

# Evaluate the best svm on test set
y_test_pred,scores_test = svm.predict(X_test)
print("test score is:",scores_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)
test_accuracy_list.append(test_accuracy)
train_accuracy_list.append(train_accuracy)
val_accuracy_LIST.append(val_accuracy)

# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength, these may
# or may not be nice to look at.
print(test_accuracy_list)
print(train_accuracy_list)
print(val_accuracy_LIST)
plt.figure(2)
plt.subplot(1,3,1)  
plt.plot(test_accuracy_list)
plt.xlabel('fitting number')
plt.ylabel(' test accuracy')

plt.subplot(1,3,2)  
plt.plot(train_accuracy_list)
plt.xlabel('fitting number')
plt.ylabel('train accuracy')


plt.subplot(1,3,3)  
plt.plot(val_accuracy_LIST)
plt.xlabel('fitting number')
plt.ylabel(' validation accuracy')

plt.show()


plt.figure(2)
plt.subplot(1,3,1)  
plt.plot(grad_numerical_list)
plt.xlabel('Iteration number')
plt.ylabel('gradient numerical')

plt.subplot(1,3,2)  
plt.plot(grad_analytic_LIST)
plt.xlabel('Iteration number')
plt.ylabel('gradient _analytic')


plt.subplot(1,3,3)  
plt.plot(rel_error_list)
plt.xlabel('Iteration number')
plt.ylabel('real error')

plt.show()
''''''''''
  w = svm.W[:-1,:] # strip out the bias
  w = w.reshape(input_width, input_height, 3, -1)
  print("size of w :",np.shape(w))
  w_min, w_max = np.min(w), np.max(w)
  #classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  for i in range(30):
      plt.subplot(2, 50, i + 1)
        
      # Rescale the weights to be between 0 and 255
      wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
      plt.imshow(wimg.astype('uint8'))
      plt.axis('off')
      plt.title(classes[i])
  plt.show()    
  '''''''''''