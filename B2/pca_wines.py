import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from numpy import cov



def PCA(data,n_components): #Compute Eigen values and Eigen vectors

    data_file = np.loadtxt(data)
    D, V = eig(cov(data_file, rowvar=False))
    V = V[:, np.argsort(D)] #Eigen vector with maximum variance is at lastt

    #eigen values are not very similar there is variation so the dataset is not in a good subspace

    X = data_file - np.tile(data_file.mean(axis=0), [data_file.shape[0], 1])
    D = sorted(D ,reverse=True)

    var=[]

    for k in range(n_components):

        var.append(D[k])

    #Find info given by first component

    print "Variance",str(var/sum(D)*100)+'%'
    print "Component 1",V[:,-1]
    print "Component 2",V[:,-2]
    # Map X through the PCA (rotate axes):

    X_rot = np.matmul(X, V)
    transformed = X_rot[:,:n_components]   #2 pca

    #project the data onto these transformation             #check value of quality column in origional file
    low=[]
    medium=[]
    high=[]
    labels=["Low quality","Medium quality","High quality"]
    for i in range(data_file.shape[0]):

        if data_file[i][11]<=5:
            #plt.scatter(transformed[i][0],transformed[i][1],color='red')
            low.append(i)
        elif data_file[i][11]==6:
            #plt.scatter(transformed[i][0], transformed[i][1], color='blue')
            medium.append(i)
        elif data_file[i][11] >=7:
            #plt.scatter(transformed[i][0], transformed[i][1], color='green')
            high.append(i)

    plt.scatter(transformed[low,0],transformed[low,1],label=labels[0],color='red')
    plt.scatter(transformed[medium,0], transformed[medium,1], label=labels[1], color='blue')
    plt.scatter(transformed[high][0], transformed[high][0], label=labels[2], color='green')

    plt.legend(loc='lower left',
           ncol=3,
           fontsize=10)
    plt.title(data)
    plt.show()

def VariableRanking(data):
    #one variable per projection

    data_file = np.loadtxt(data)
    variable_num=0
    Eigen=[]
    zero_columns=[]
    for i in range(data_file.shape[1]):
        data_file = np.loadtxt(data)
        zero_columns=[x for x in range(data_file.shape[1])if x!=i ]
        data_file[:,zero_columns]=0
        D, V = eig(cov(data_file, rowvar=False))
        a=(D[i],i)
        Eigen.append(a)

    low=[]
    medium=[]
    high=[]

    V=sorted(Eigen,key=lambda eigen_value: eigen_value[0],reverse=True)
    print V #Eigen vectors and their values
    print V[0][1]
    print V[1][1]
    labels = ["Low quality", "Medium quality", "High quality"]

    for i in range(data_file.shape[0]):

        if data_file[i][11]<=5:
            #plt.scatter(transformed[i][0],transformed[i][1],color='red')
            low.append(i)
        elif data_file[i][11]==6:
            #plt.scatter(transformed[i][0], transformed[i][1], color='blue')
            medium.append(i)
        elif data_file[i][11] >=7:
            #plt.scatter(transformed[i][0], transformed[i][1], color='green')
            high.append(i)

    plt.scatter(data_file[low,6], data_file[low, 5], label=labels[0], color='red')
    plt.scatter(data_file[medium, 6], data_file[medium, 5], label=labels[1], color='blue')
    plt.scatter(data_file[high][6], data_file[high][5], label=labels[2], color='green')

    plt.legend(loc='lower left',
               ncol=3,
               fontsize=10)



    plt.title("Variable Ranking")
    plt.show()

def Main():

    file1="winequality-red.txt"
    file2="winequality-white.txt"
    n_components=2
    PCA(file2,n_components)
    VariableRanking(file2)


Main()