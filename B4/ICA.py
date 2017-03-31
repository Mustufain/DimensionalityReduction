#Independent Component Analysis
# 4 audio independent sources

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.io import wavfile
from scipy import signal
from pylab import imread,subplot,imshow,title,gray,figure,show,NullLocator
import plotly.plotly as py


def ReadAudioFile():
    sample_freq,  data=wavfile.read("ica_mixdown01.wav")
    sample_freq1, data1 = wavfile.read("ica_mixdown02.wav")
    sample_freq2, data2 = wavfile.read("ica_mixdown03.wav")
    sample_freq3, data3 = wavfile.read("ica_mixdown04.wav")

    time = np.linspace(0, data.shape[0] / sample_freq, num=data.shape[0])



    timeseries = np.zeros(shape=(data.shape[0],4))

    for i in range(data.shape[0]):

        timeseries[i][0] = (data[i])   * (time[i])
        timeseries[i][1] = (data1[i])  * (time[i])
        timeseries[i][2] = (data2[i]) * (time[i])
        timeseries[i][3] = (data3[i])  * (time[i])



    return timeseries,sample_freq,time

def Independent_Component_Analysis(timeseries,freq,time):


    ica = FastICA(n_components=4)
    S_=ica.fit_transform(timeseries)


    wavfile.write("test1.wav",freq,S_[:,0]*4)
    wavfile.write("test2.wav", freq, S_[:, 1] * 4)
    wavfile.write("test3.wav", freq, S_[:, 2] * 4)
    wavfile.write("test4.wav", freq, S_[:, 3] * 4)

    plt.figure(1)

    ay=plt.subplot(211)
    ay.set_title("All waveforms")
    ay.plot(time,timeseries)

    ax=plt.subplot(212)
    ax.set_title("Train noise removed")
    ax.plot(time,S_[:,[0,2,3]])

    plt.show()

def SeperateComponents():  #This is the c part

    sample_freq, data = wavfile.read("ica_mixdown05.wav")
    sample_freq1, data1 = wavfile.read("ica_mixdown06.wav")
    sample_freq2, data2 = wavfile.read("ica_mixdown07.wav")
    sample_freq3, data3 = wavfile.read("ica_mixdown08.wav")

    time = np.linspace(0, data.shape[0] / sample_freq, num=data.shape[0])

    timeseries = np.zeros(shape=(data.shape[0], 4))

    for i in range(data.shape[0]):
        timeseries[i][0] = (data[i]) * (time[i])
        timeseries[i][1] = (data1[i]) * (time[i])
        timeseries[i][2] = (data2[i]) * (time[i])
        timeseries[i][3] = (data3[i]) * (time[i])

    #print timeseries.shape
    ica = FastICA(n_components=4)
    S_ = ica.fit_transform(timeseries)

    wavfile.write("test5.wav", sample_freq, S_[:, 0] * 50)
    wavfile.write("test6.wav", sample_freq, S_[:, 1] * 50)
    wavfile.write("test7.wav", sample_freq, S_[:, 2] * 50)
    wavfile.write("test8.wav", sample_freq, S_[:, 3] * 50)






def Main():
    timeseries,freq,time = ReadAudioFile()
    Independent_Component_Analysis(timeseries,freq,time)
    #SeperateComponents()       #part c

Main()