## Nonnegative Matrix Factorization
import numpy as np
import sklearn.datasets as ds
iris=ds.load_iris()
X=iris.data
XT=X.T.copy() ## each column is an instance
K=3 ## number of components (clusters)

C1=np.random.choice(K,XT.shape[1])  ## initial clustering (randomly)
### or initialize with the real clusters
#C1[0:50]=0
#C1[50:100]=1
#C1[100:]=2
##


wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra
data=np.loadtxt(open("pendigits-1.csv", "r"), delimiter=",")
X=data[:,:-1]/100
from sklearn.cluster import KMeans
Km = KMeans(n_clusters=2).fit(X)

C1=Km.predict(X)
C2=MNMF(C1,X.T,2,.5)

## objective function
def objfct(X,V,U,S,lmbd=.2):
  VUT=V.dot(U.T)
  return np.trace(X.dot(X.T)) - 2 * np.trace(X.dot(VUT))+np.trace(U.dot(V.T).dot(VUT))+lmbd*np.trace(V.T.dot(S).dot(V))

## build similarity matrix
def buildSM(S,N,C1):
  for i in range(N):
    for j in range(i,N):
      if C1[i]==C1[j]:
         S[i,j]=1 #xi and xj are in the same cluster
  S=np.bitwise_or(S,S.T)
  return S

#Multiple NMF
# C1 initial clustering
# X datapoints: rows: features, columns: instances
#    X4x100: 100 examples, 4 features
# K number os clusters
# threshold for the residual between successive objectives
# Lambda
def MNMF(C1,X,K,thr=1,lmbd=.2):
  print('Starting MNMF...')
  N=X.shape[1] ## number of instances
  M=X.shape[0] ## number of features
  S=np.zeros((N,N)).astype(int) ## Similarity matrix (initialize with 0)
  S=buildSM(S,N,C1)
  np.random.seed(1)
  U=np.random.rand(M,K) ##  feature matrix
  V=np.random.rand(N,K) ## coeficient matrix
  objbf=objfct(X,V,U,S,lmbd)
  rsd=thr+1 #(the residual is bigger than the threshold)
  c=1
  while (rsd>thr):
    ## update U
    XV=X.dot(V)
    UVTV=U.dot(V.T).dot(V)
    d=XV/UVTV
    U=U*d
    ## update V
    XTU=X.T.dot(U)
    VUTU=V.dot(U.T).dot(U)
    LSV=lmbd*(S.dot(V))
    d=XTU/(VUTU+LSV)
    V=V*d
    ## normalize U
    for k in range(K):
       for i in range(M):
           U[i,k]=U[i,k]/np.sqrt((U[:,k]**2).sum()) 
    ## normalize V
    for k in range(K):
      for j in range(N):
         V[j,k]=V[j,k]*np.sqrt((V[:,k]**2).sum())
    objaf=objfct(X,V,U,S,lmbd)
    rsd=np.abs(objbf-objaf)  ### difference between objectives
    print(c,'Objectives: before(',objbf,')','after(',objaf,')','diff:',rsd)
    c=c+1
    objbf=objaf
  ### new clustering
  C2=np.argmax(V,1)
  return C2
