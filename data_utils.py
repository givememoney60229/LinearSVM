from six.moves import cPickle as pickle
import numpy as np
import os
#from scipy.misc import imread
import platform
import cv2
from scipy import sparse
from sklearn.decomposition import PCA

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


def load_data(filename):
  fh=open(filename,"r",encoding="utf-8")
  lines=fh.readlines()
  data=[]
  label=[]
  for line in lines:
      line=line.strip("\n")
      line=line.strip()
      words=line.split()
      imgs_path=words[0]
      labels=words[1]
      label.append(labels)
      data.append(imgs_path)
  return data,label 

def low_rank_approx(SVD=None, A=None, r=1):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD
    Requires: numpy
    """
    if not SVD:
        SVD = np.linalg.svd(A, full_matrices=False)
    u, s, v = SVD
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar
def compute_basis(X,N):
        x_sparse=sparse.csr_matrix(X).asfptype()
        id=x_sparse@x_sparse.transpose()
        _,vecs = sparse.linalg.eigsh(id, k=N,which='LM')
        return vecs


def load_mydata(filename,width,height,profect=False):
 
  data,label=load_data(filename)
  xs = []
  ys = []
  for i in range(len(label)):
    image_dir="C:/Users/user/Desktop/"
    img_path=os.path.join(image_dir,data[i])
    image=cv2.imread(img_path)
    
    if image.ndim==2:
        image=cv2.cvtColor(image,cv2.COLOR_BAYER_BG2BGR)
    X=cv2.resize(image,(width, height), interpolation=cv2.INTER_AREA)
    if profect==True:
      X_2d=X.transpose(0,2,1).reshape(3,-1)
      E=compute_basis(X_2d,1)
      X=np.matmul(E.transpose(),X_2d)
      #X=low_rank_approx(SVD=True, A=X.transpose(0,2,1).reshape(3,-1), r=3)
      X=X.transpose(0,1).reshape(-1,width,height).transpose(0,2,1)
    
    
    xs.append(X)
    
    


  Xtr = np.array(xs)
  Ytr = np.asarray(label,dtype=int)
  
  return Xtr, Ytr  