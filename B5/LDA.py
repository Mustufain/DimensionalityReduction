#Applying Linear Discriminant Analysis for Dimensionality Reduction


import numpy as np
from sklearn.datasets import  fetch_olivetti_faces
from numpy.linalg import eig
from numpy import cov
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.lda import LDA
from sklearn.decomposition import PCA

#4096 features

class_label=0

def LoadData():

    dataset = fetch_olivetti_faces()
    faces = dataset.data
    labels =  dataset.target
    n_samples, n_features = faces.shape
    label_set=list(set(labels))
    classes_list=[]
    faces = faces - np.tile(faces.mean(axis=0), [faces.shape[0], 1])
    for i in range(len(label_set)):
        indices=[]
        for j in range(len(labels)):
            if labels[j] == label_set[i]:
                indices.append(j)
        classes_list.append(faces[indices,:])

    return classes_list,faces,labels



def LinearDiscriminantAnalysis(classes_list,faces,n_components):
    Sb=0
    Sw = 0
    Mean = 0
    #Each class has 10 samples and 4096 features

    c=0
    classMean=[]



    for i in classes_list:
        classMean.append(np.mean(i,axis=0))
        Mean += np.mean(i, axis=0)


    for i in classes_list:  # First i is np array of 0 class

        Sb+=np.multiply((len(i)),np.outer((classMean[c]-Mean),(classMean[c]-Mean)))

        c+=1

    #The within-class scatter matrix
    Sw = np.zeros(Sb.shape)
    sw=0
    c=0
    column=0
    for i in classes_list:  #i is np array of zero class

        Sw+=np.cov(i.T)


    #S_{w}^{-1}S_{b}w = \lambda w




    EigenMatrix = np.dot(np.linalg.inv(Sw), Sb)
    D, V = np.linalg.eigh(EigenMatrix)


    V = V[:, np.argsort(D)]  # Eigen vector with maximum variance is at last
    D = sorted(D, reverse=True)


    W=np.column_stack((V[:,-1],V[:,-2])) #(4096,2)  Combine 2 highest eigen vectors
    X = faces - np.tile(faces.mean(axis=0), [faces.shape[0], 1])
    Transformed = np.matmul(X,W) #(400,4096)(4096,2)  #Project the data onto 2 highest eigenvectors


    # Analyze Projection matrix
    # Ranking of eigen values gives importance of each feature




    return Transformed  #(400,2)



def ScikitLDA():

    clf=LDA(n_components=2)

    dataset = fetch_olivetti_faces()
    faces = dataset.data
    labels = dataset.target

    Transformed = clf.fit(faces,labels).transform(faces)
    print clf.coef_.shape
    return Transformed




def Visualization(transformed):

    labels=[]
    for i in range(transformed.shape[0]/10):
        labels.append(str(i)) #40 colors
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    labelcount=-1

    for k in range(0,transformed.shape[0],10):

        labelcount+=1
        plt.scatter(transformed[k:k+10,0],transformed[k:k+10,1],label=labels[labelcount],color=colors[labelcount])

    plt.legend(loc='lower left',
           ncol=3,
           fontsize=7)

    plt.show()










def Main():
    n_components=2
    classes_list,faces,labels=LoadData()
    transformed=LinearDiscriminantAnalysis(classes_list,faces,n_components)
    transformed=ScikitLDA()
    Visualization(transformed)


Main()