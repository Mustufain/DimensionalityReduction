#Forward selection
import numpy as np
import operator
import matplotlib.pyplot as plt


data = np.loadtxt('noisy_sculpt_faces.txt')


def NearestNeighbor(face_i,data,index,n_features):
    row = face_i
    SSE=[]
    index=index

    for i in range(len(data)):

        if index!=i: #Exclude the ith face as it is leave one out


            sse=(i,np.sum((row[n_features]-data[i][n_features])**2))
            #sse=(i,np.sum((row[n_features]-data[i][n_features])**2))
            SSE.append(sse)


    SSD = min(SSE, key = lambda error : error[1])

    return SSD

def Predict(SSD,data,index):
    predict_label=[]

    for i in range(len(data)):

        if i == SSD[0]:

            target_predict=(index,data[i][256:])
            predict_label.append(target_predict)
    return predict_label

def Sum_Of_Squared_Differences(predict_label):

    result =0
    for i in predict_label:
        index=i[0]
        predict_value = i[1]
        actual_value = data[index][256:]

    for i in range(3):

        e = (actual_value[i]-predict_value[i])**2 + result
        result = e

    return (index,e)

def ForwardSelection(Features,data):

    #Find variance of each column
    #Increase features uptil there is no change in SSE
    #Features 256

    temp_chosen_feature=[]
    error_list=[]
    chosen_feature=[]
    Feature_count=[]
    tempset=[]
    count=0
    for i in range(Features):
        Feature_count.append(i)

    for i in range(Features):           #it would run 256 times


        for k in Feature_count:

            tempset.extend(chosen_feature)
            tempset.append(k)
            index=0
            total=0

            for face_i in data:


                SSD=NearestNeighbor(face_i,data,index,tempset)
                target_pred=Predict(SSD,data,index)
                error=Sum_Of_Squared_Differences(target_pred)
                total = total + error[1]
                index+=1


            total=total/100
            e=(k,total)
            error_list.append(e)
            tempset=[]
        count+=1
        min_sse = min(error_list, key=lambda error: error[1]) #we get a tuple (feature,sse)
        chosen_feature.append(min_sse[0])
        m=(chosen_feature,min_sse[1])
        temp_chosen_feature.append(m)
        Feature_count.remove(min_sse[0])
        error_list=[]


    best_feature = min(temp_chosen_feature, key=lambda sse: sse[1])  # Feature set with min SSE
    print best_feature
    Feature_set = list(zip(*temp_chosen_feature)[0])
    F = [i for i in range(0,256) ]
    SSE = list(zip(*temp_chosen_feature))[1]

    plt.xlabel("Features")
    plt.ylabel("SSE")
    plt.plot(F,SSE)
    plt.show()
    return error_list


def Visualization(error_list):

    plt.xlabel("Features")
    plt.ylabel("SSE")
    plt.axis([0,255,0,10000])
    plt.plot(error_list)
    plt.show()

#def VariableRanking():




def Main():
    index=0
    total=0
    #for face_i in data:

        #SSD = NearestNeighbor(face_i,data,index,[x for x in range(0,256)])
        #target_pred=Predict(SSD,data,index)
        #error = Sum_Of_Squared_Differences(target_pred)
        #total = total + error[1]
        #index += 1

    #error_list = ForwardSelection(256,data)
    #Visualization(error_list)
    #print (total/100)



Main()