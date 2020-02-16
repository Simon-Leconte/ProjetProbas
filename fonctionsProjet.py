import numpy as np
import math
import matplotlib.pyplot as plt


A=0
B=500
N=101
Delta= (B-A)/(N-1)
discretization_indexes = np.arange(N)
discretization = discretization_indexes*Delta

mu=-5
a=50
sigma2=12

observation_indexes = [0,20,40,60,80,100]
depth = np.array([0,-4,-12.8,-1,-6.5,0])

unknown_indexes=list(set(discretization_indexes)-set(observation_indexes))

### Question 1
def covariance(d,a,v):
    return v*np.exp(-abs(d)/a)


### Question 2
def distance(discretization):
    n=len(discretization)
    D=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,j]=abs(discretization[i]-discretization[j])
    return D

distance_matrix=distance(discretization)

###Question 3

def extraction(M,lines,columns):
    #cette fonction est aussi utilisée dans la question 4
    L=M[lines]   #L est la matrice des lignes qu'on veut extraire
    n=len(lines)
    m=len(columns)
    N=np.zeros((n,m))
    for i in range(m):
        N[i]=L[i][columns]
    return N

def gauss(unknown_indexes):
    n=len(unknown_indexes)
    return np.random.normal(0,1,n)

def cholesky(A):
    """Renvoie une matrice B telle que B.B^T=A """
    return np.linalg.cholesky(A)

def simulation(mu,sigma2,a,unknown_indexes):
    n=len(unknown_indexes)
    Y=gauss(unknown_indexes)
    M=mu*np.ones(n)
    C=covariance(extraction(distance_matrix,unknown_indexes,unknown_indexes),a,sigma2)
    R=cholesky(C)
    return M+R.dot(Y)



###Question 4


def ext_cov_observations(M,observation_indexes):
    """Extrait la matrice de covariance entre les observations à partir de M """
    return extraction(M,observation_indexes,observation_indexes)


def ext_cov_observations_unknown(M,observation_indexes,unknown_indexes):
    """Extrait la matrice de covariance entre les observations et les inconnues à partir de M """
    return extraction(M,observation_indexes,unknown_indexes)

def ext_cov_unknown(M,unknown_indexes):
    """Extrait la matrice de covariance entre les inconnues à partir de M """
    return extraction(M,unknown_indexes,unknown_indexes)

###Question 5


