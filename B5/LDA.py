#Applying Linear Discriminant Analysis for Dimensionality Reduction


import numpy as np
from sklearn.datasets import  fetch_olivetti_faces
from numpy.linalg import eig
from numpy import cov
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
    for i in classes_list:
        Mean += np.mean(i, axis=0)

    for i in classes_list:  # First i is np array of 0 class

        ClassMean = np.mean(i, axis=0)

        Sb += np.multiply.outer(ClassMean - Mean, ClassMean - Mean)

        Sw += cov(i, rowvar=False)

    Sb = Sb / len(classes_list)
    # print Sb.shape
    # print Sw.shape

    EigenMatrix = np.matmul(np.linalg.inv(Sw), Sb)
    D, V = eig(EigenMatrix)
    # print w1.shape
    V = V[:, np.argsort(D)]  # Eigen vector with maximum variance is at last
    # Transofrm data to 2 dimensions

    X_rot = np.matmul(faces, V)
    transformed = X_rot[:, :n_components]  # 2 dimension
    # print transformed.shape
    return transformed

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




#Analyze Projection matrix








def Main():
    n_components=2
    classes_list,faces,labels=LoadData()
    transformed=LinearDiscriminantAnalysis(classes_list,faces,n_components)
    Visualization(transformed)



Main()