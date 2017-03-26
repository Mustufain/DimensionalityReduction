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

    error_list=[]
    chosen_feature=[]
    Feature_count=[]
    tempset=[]
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

            tempset=[]

            total=total/100
            e=(k,total)
            error_list.append(e)

        min_sse = min(error_list, key=lambda error: error[1]) #we get a tuple (feature,sse)

        chosen_feature.append(min_sse[0])

        Feature_count.remove(min_sse[0])
        error_list=[]
        print min_sse
        break





        #Feature_set.append(min_sse[0])
        #Features.remove(min_sse)

        #Find minimum error and which feature also

        #Feature_set=[]
        #col_var = np.var(data[:,i])
        #v = (i,col_var)
        #var.append(v)



 #   sorted_var = sorted(var, key=lambda variance: variance[1])

    #for i in range(len(var)): #first column is of maximum variance

        #Feature_set.append(var[i][0])
        #index=0
        #total=0
        #for face_i in data:

            #SSD =  NearestNeighbor(face_i,data,index,Feature_set)
            #target_pred = Predict(SSD,data,index)
            #error = Sum_Of_Squared_Differences(target_pred)
            #total = total + error[1]
            #index+=1
        #error_list.append(total/100)
        #print Feature_set
        #Feature_set=[]
    return error_list

        #print (Feature_set,len(Feature_set),total/100)

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

        #SSD = NearestNeighbor(face_i,data,index,256)
        #target_pred=Predict(SSD,data,index)
        #error = Sum_Of_Squared_Differences(target_pred)
        #total = total + error[1]
        #index += 1

    error_list = ForwardSelection(256,data)
    #Visualization(error_list)

   # print (total/100)



Main()