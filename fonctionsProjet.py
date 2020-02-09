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
