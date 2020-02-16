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

#Question 1
def covariance(d,a,v):
    return v*np.exp(-np.abs(d)/a)


#Question 2
def distance(discretization):
    n=len(discretization)
    D=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i][j]=np.abs(discretization[i]-discretization[j])
    return D

distance_matrix=distance(discretization)

#Question 3

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

C=covariance(distance_matrix,a,sigma2)

#Question 4


def ext_cov_observations(M):
    """Extrait la matrice de covariance entre les observations à partir de M """
    return extraction(M,observation_indexes,observation_indexes)


def ext_cov_observations_unkuwns(M):
    """Extrait la matrice de covariance entre les observations et les inconnues à partir de M """
    return extraction(M,observation_indexes,unknown_indexes)

def ext_cov_unkuwns(M):
    """Extrait la matrice de covariance entre les inconnues à partir de M """
    return extraction(M,unknown_indexes,unknown_indexes)

#Question 5

"""def loi_condi_sachant_observation(x,y):
    C=covariance(distance_matrix,a,sigma2)
    CX=ext_cov_unkuwns(C)
    CY=ext_cov_observations(C)
    CXY= ext_cov_observations_unkuwns(C)
    CSx = CX - (CXY.dot(numpy.linalg.inv(CY))).dot(CXY)
    n=len(y)
    Psi=mu*np.ones(N-n+1)- (CXY.dot(numpy.linalg.inv(CY))).dot(y-u*np.ones(N-n+1))
    det=numpy.linalg.det(CSx)
    return (np.exp((-0.5*np.transpose(x-psi).dot(numpy.linalg.inv(CSx))).dot(x-psi)))/(((2*np.pi)**((N-n+2)/2))*((det)**0.5))""" #C'est la methode de calcul directe saans utiliser la 3 question théorique

def esp_cond(depth,mu):
    n=len(observation_indexes)
    C=covariance(distance_matrix,a,sigma2)
    R=cholesky(C)
    R_observation=extraction(R,observation_indexes,observation_indexes)
    standard_normal_observation=np.linalg.solve(R_observation,depth-np.array([mu]*n)) #C'est les valeurs de Y[i] (pour i dans observation_indexes) pour lesquelles Z[i]=l'observation
    Y=np.array([0.0]*N)
    for i in range(n):
        z[observation_indexes[i]]=standard_normal_observation[i]
    return np.array([mu]*N)+R.dot(Y)

esp=esp_cond(depth,mu)

plt.plot(discretization,esp)
plt.show()






