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

    print "Variance",var
    print "Component 1",V[:,-1]
    print "Component 2",V[:,-2]
    # Map X through the PCA (rotate axes):

    X_rot = np.matmul(X, V)
    transformed = X_rot[:,:2]   #2 pca

    #project the data onto these transformation
    plt.scatter(transformed[:,0],transformed[:,1])
    plt.title(data+":" + "Data plotted on 2 principal components")
    plt.show()

def VariableRanking(data):
    #one variable per projection
    var=[]
    data_file=np.loadtxt(data)
    for i in range(data_file.shape[1]):
         a=(i,np.var(data_file[:,i]))
         var.append(a)

    var = sorted(var ,key=lambda variance:variance[1],reverse=True)

    for i in range(2):
        print var[i]

    plt.scatter(data_file[:,6],data_file[:,5])
    plt.title("Variable Ranking")
    plt.show()

def Main():

    file1="winequality-red.txt"
    file2="winequality-white.txt"
    n_components=2
    PCA(file2,n_components)
    VariableRanking(file2)
    #redwine_data,red_components=red_wines()
    #whitewine_data,white_components = white_wines()
    #ScatterPlotRedWine(redwine_data,red_components)
    #ScatterPlotWhiteWine(whitewine_data,white_components)


Main()