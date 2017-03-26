#Quesion 1 of exercise dimensionality reduction

import numpy as np
#mean+- SD is the range
def GenerateDataPoint(dim):
    points = 10000000
    data = np.empty(shape=(dim,points))
    for i in range(dim):
        data[i] = np.random.uniform(1,-1,points)




    pointOutsideHypershpere=0
    pointInsideHypersphere=0
    pointsInsideHyperSphericalShell=0
    for i in range(points):

            distance = np.linalg.norm(data[:,i]) #Euclidean Distance

            if distance <= 1 :

                pointInsideHypersphere  += 1

            if 0.95<distance and distance<1:

                pointsInsideHyperSphericalShell+=1

            else:

                pointOutsideHypershpere+=1

    print ("Dimension: " + str(dim),str(float(pointInsideHypersphere)/float(points) * 100)+"%","inside hemisphere")
    print ("Dimension: " + str(dim), str(float(pointsInsideHyperSphericalShell) / float(points) * 100) + "%", "inside HyperShpericalShell")



for i in range(3,15,3):
    GenerateDataPoint(i) #Dimension 3,6,9,12

