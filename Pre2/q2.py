import numpy as np
import random
import operator
import matplotlib.pyplot as plt

def GenerateDataPoints(dim):
    points=2000
    dataset=[]
    data = np.empty(shape=(dim,points))
    for i in range(dim):
        data[i] = np.random.normal(0,1,points)

    for i in data[0]:  #Taking coordinates of each data point of first dimension
        target = np.sin(i)
        t = (i,target)
        dataset.append(t)


    #Split dataset 1000 Training and 1000 testing

    train_data =  data[:,:points/2] #First 1000 columns
    test_data = data[:,points/2:points]  #Last 1000 columns
    return train_data,test_data,points,dataset

def getNeighbors(train_data,test_data,k,points):
    k=5
    Euclidean = []
    Neighbors=[]
    Neighbors_x_test=[]
    for x in range(0,points/2):
        dist=(train_data[0][x],np.linalg.norm(train_data[:,x]-test_data))
    #for j in range(0, points / 2):
        #for i in range(0, points / 2):
            #dist = (train_data[0][i], np.linalg.norm(train_data[:, i] - test_data[:, j]))
        Euclidean.append(dist)

    Euclidean.sort(key=operator.itemgetter(1))
    for i in range(k):

        Neighbors.append(Euclidean[i][0])




    return test_data[0],Neighbors

def Predict(Neighbors,x_test,dataset):
    sum=0
    x_test = x_test
    for i in Neighbors:
        for j in dataset:
            if i == j[0]:

                result = sum + j[1]
                sum = result
                break
    predicted = float(result)/float(len(Neighbors))



    for i in dataset:
        if x_test == i[0]:
            target_actual = i[1]
            break

    return predicted,target_actual

def Visualization(pred_actual):
    plt.xlabel("Data cordinate x1 in test_set")
    plt.ylabel("Target variable")


    plt.plot(list(zip(*pred_actual)[0]),list(zip(*pred_actual)[2]),'r') #Actual
    plt.plot(list(zip(*pred_actual)[0]),list(zip(*pred_actual)[1]),'b') #Predicted


    plt.show()


def Error(pred_actual,points):

    sum=0
    pred = list(zip(*pred_actual)[2])
    actual = list(zip(*pred_actual))[1]

    for x in range(len(pred_actual)):
        SSE = (actual[x] - pred[x])**2
        sum = SSE


    return float(float(SSE)/points)

def Main():

    pred_actual=[]
    train_data,test_data,points,dataset = GenerateDataPoints(6)

    for x in range(0,points/2):

        x_test,Neighbors = getNeighbors(train_data,test_data[:,x],5,points)
        Predicted,Actual=Predict(Neighbors,x_test, dataset)
        result = (x_test,Predicted,Actual)
        pred_actual.append(result)
        if x==10:
            break
        #print ("Predicted: ", str(Predicted),"Actual: ",str(Actual))



    Visualization(pred_actual)
    print Error(pred_actual,points)

Main()


