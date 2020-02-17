import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


A=0.0
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

C=covariance(distance_matrix,a,sigma2)


#Question 4

def extraction(M,lines,columns):
    #cette fonction est aussi utilisée dans la question 4
    L=M[lines]   #L est la matrice des lignes qu'on veut extraire
    n=len(lines)
    m=len(columns)
    T=np.zeros((n,m))
    for i in range(n):
        T[i]=L[i][columns]
    return T

def ext_cov_observations(M):
    """Extrait la matrice de covariance entre les observations à partir de M """
    return extraction(M,observation_indexes,observation_indexes)

CY=ext_cov_observations(C)

def ext_cov_observations_unkuwns(M):
    """Extrait la matrice de covariance entre les observations et les inconnues à partir de M """
    return extraction(M,observation_indexes,unknown_indexes)

CYX=ext_cov_observations_unkuwns(C)

CXY=np.transpose(CYX)

def ext_cov_unkuwns(M):
    """Extrait la matrice de covariance entre les inconnues à partir de M """
    return extraction(M,unknown_indexes,unknown_indexes)

CX=ext_cov_unkuwns(C)

#Question 5

n=len(depth)
esp_cond_unkown_points=mu*np.ones(N-n)+ (CXY.dot(np.linalg.inv(CY))).dot(depth-mu*np.ones(n)) #on a directement la formule de l'esperence


esp=np.zeros(N)
j=0
for i in range(N):
    if i in observation_indexes:
        esp[i]=depth[j]
        j+=1
    else:
        esp[i]=esp_cond_unkown_points[i-j]




#plt.plot(discretization,esp)
#plt.show()

#Question 6
def covariance_cond_all(depth,mu):
    return CX - np.dot(CXY,np.dot(np.linalg.inv(CY),CYX))

def var_cond(depth,mu):
    cov=covariance_cond_all(depth,mu)
    var=np.zeros(N)
    j=0
    for i in range(N):
        if i in observation_indexes:
            j+=1
        else:
            var[i]=cov[i-j][i-j]
    return var


var=var_cond(depth,mu)


#plt.plot(discretization,var)
#plt.show()


#Question7

R = np.linalg.cholesky(covariance_cond_all(depth,mu))

def simulation(mu,sigma2,a,unknown_indexes,depth):
    Y = np.random.normal(0,1,N-n)
    sim = esp_cond_unkown_points + R.dot(Y)
    sim_all = np.zeros(N)
    j=0
    for i in range(N):
        if i in observation_indexes:
            sim_all[i] = depth[j]
            j+=1
        else:
            sim_all[i]=sim[i-j]
    return sim_all

simu = simulation(mu,sigma2,a,unknown_indexes,depth)

#plt.plot(discretization,simu)
#plt.show()


#Question8

def length(depth_all,Delta):
    length=0
    for i in range(1,N):
        z = depth_all[i]-depth_all[i-1]
        length+=np.sqrt(Delta**2+z**2)
    return length


#Question9

length_list = np.array([length(simulation(mu,sigma2,a,unknown_indexes,depth),Delta) for i in range(100)])

length_expected_value = np.average(length_list)
length_cond_expected_value = length(esp , Delta)

print("la longueur du câble à partir de 100 simulations : ", length_expected_value)
print("l’espérance conditionnelle de la longueur avec la longueur de l’espérance conditionnelle : ",length_cond_expected_value)
print("la différence entre les deux est : ", np.abs(length_expected_value-length_cond_expected_value))
print("la différence relative entre les deux est : ", (np.abs(length_expected_value-length_cond_expected_value)/min(length_cond_expected_value,length_expected_value))*100,"%")

#on remarque toujours que la difference relative est de l'ordre de 0,04%


#Question10

def M(n):
    expected_length = 0
    for i in range(n):
        expected_length += length(simulation(mu,sigma2,a,unknown_indexes,depth),Delta)/n
    return expected_length

simulations_number = 100

simulations_number_list = [i+1 for i in range(simulations_number)]
length_list_n = np.array([M(i+1) for i in range(simulations_number)])

#plt.plot(simulations_number_list,length_list_n)
#plt.show()

relative_error = (length_list_n/length_cond_expected_value - np.array([1.0]))*100

#C'est l'erreur relative (en pourcentage) par rapport à l’espérance conditionnelle de la longueur avec la longueur de l’espérance conditionnelle

#plt.plot(simulations_number_list,relative_error)
#plt.show()


#Question11


df = pd.DataFrame(length_list_n,columns = ['longueur du câble'])
res = df.plot.hist(bins=50)
print(res)
#plt.show()


#Question12

tolerance=0.95

   #methode1


   #methode2

#Question13

total_number = np.sum(res[0])
max_length = np.max(res[1])
k=len(res[0])
prob=0
for i in range(k):
    if res[1][i] > 525:
        prob+=res[0][i]/total_number

print("une estimation de la probabilité que la longueur du câble dépasse 525 m est : ",prob)