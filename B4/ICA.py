#Independent Component Analysis
# 4 audio independent sources

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.io import wavfile
from scipy import signal


def ReadAudioFile():
    sample_freq,  data=wavfile.read("ica_mixdown01.wav")
    sample_freq1, data1 = wavfile.read("ica_mixdown02.wav")
    sample_freq2, data2 = wavfile.read("ica_mixdown03.wav")
    sample_freq3, data3 = wavfile.read("ica_mixdown04.wav")

    time = np.linspace(0, data.shape[0] / sample_freq, num=data.shape[0])


    #plt.title("Mixed audio Signal wave...")
    #plt.plot(timearray,data)
    #plt.xlabel("Time/s")
    #plt.ylabel("Amplititude")
    #plt.show()

    #time series data set

    timeseries = np.zeros(shape=(data.shape[0],4))

    for i in range(data.shape[0]):

        timeseries[i][0] = (data[i])   * (time[i])
        timeseries[i][1] = (data1[i])  * (time[i])
        timeseries[i][2] = (data2[i]) * (time[i])
        timeseries[i][3] = (data3[i])  * (time[i])

    #Add Noise

    timeseries += timeseries * np.random.normal(size=timeseries.shape)
    timeseries /= timeseries.std(axis=0)  # Standardize data

    # Mix data
    A = np.array([[1, 1, 1,1], [0.5, 2, 1.0,1.5], [1.5, 1.0, 2.0,2.5]])  # Mixing matrix
    timeseries = np.dot(timeseries, A.T)  # Generate observations

    plt.title("with noise")
    plt.plot(time,timeseries[:,1])
    plt.show()
    return timeseries,sample_freq,time

def Independent_Component_Analysis(timeseries,freq,time):  #check this function


    ica = FastICA(n_components=4)
    S_=ica.fit_transform(timeseries)
    A_ = ica.mixing_
    #wavfile.write("test3.wav",freq,S_[:,1]*2)
    plt.title("without noise")
    plt.plot(time, S_[:,1])
    plt.xlabel("Time/s")
    plt.ylabel("Amplititude")
    plt.show()

def Main():
    timeseries,freq,time = ReadAudioFile()
    Independent_Component_Analysis(timeseries,freq,time)

Main()