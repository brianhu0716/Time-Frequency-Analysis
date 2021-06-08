# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 09:45:56 2021

@author: Brian Hu
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt 
from scipy.fftpack import fft
from scipy.signal import resample
import math
from scipy.interpolate import CubicSpline
import pandas as pd

class time_frequency_analyze():
    def __init__(self,data,fs) :
        self.data = data
        self.length = len(data)
        self.fs = fs
        self.time = np.arange(0,self.length / self.fs,1 / self.fs)
        '''
        plt.figure()
        plt.plot(self.time,self.data)
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.title("raw data")
        '''

    def Upsampling(self):
        if 2 ** (k := int(math.log2(self.length))) != self.length :  
            self.length = 2 ** (k + 1)
            self.data,self.time = resample(self.data,self.length,self.time)
            self.fs = self.length / (self.time[-1] - self.time[0])
        
    def FFT(self) :
        n = self.length // 2
        f_axis = np.linspace(0,self.fs // 2,n)
        spec = fft(self.data)[: n]
        plt.figure()
        plt.plot(f_axis,abs(spec))
        plt.xlabel("frequency")
        plt.ylabel("energy")
        plt.title("FFT spectrum")
        df = pd.DataFrame.from_dict({"spectrum" : spec,
                                     "frequency" : f_axis},orient = "index")
        
        return df
    def DWT(self,wavelet = None,levels = None) :
        if not wavelet :
            wavelet = pywt.Wavelet('db4')
        if not levels : 
            levels = pywt.dwt_max_level(self.length, wavelet.dec_len)
        
        coeffs = dict()
        a = self.data
        print(self.length)
        for level in range(levels + 1) :
            (cA, cD) = pywt.dwt(a, wavelet)
            print(len(cA))
            coeffs[level] = (cA, cD)
            a = cA
        '''
        for level,coeff in coeffs.items() :
            n = len(coeff[0])
            plt.figure()
            for i in range(1,3) :
                plt.subplot(2,1,i)
                if i == 1 :
                    plt.title("approximation component " + str(level))
                    plt.plot(np.linspace(0,self.time[-1],n),coeff[0])
                else :
                    plt.title("detail component " + str(level))
                    plt.plot(np.linspace(0,self.time[-1],n),coeff[1])
                plt.xlabel("time")
                plt.ylabel("amplitude")
         '''
        df = pd.DataFrame.from_dict(coeffs,orient = "index")
        return df
            
    def CWT(self,wavename = "",Nscales = None) :
        if not wavename : wavename = "morl"
        if not Nscales : Nscales = 256
        
        wc = 2 * np.pi * pywt.central_frequency(wavename)
        s0 = wc / np.pi
        NV = 16
        jmax = np.log2(self.length/2/s0/8) * NV
        scales = s0 * 2 ** (np.arange(int(jmax))/NV)
        [spectrogram, frequencies] = pywt.cwt(self.data, scales, wavename, 1 / self.fs)
        '''
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.time,self.data)
        plt.title("original data")
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.subplot(2,1,2)
        plt.contourf(self.time, frequencies, abs(spectrogram))
        plt.title("spectrogram")
        plt.xlabel("time")
        plt.ylabel("frequency")
        '''
        df = pd.DataFrame.from_dict({"spectrofram" : spectrogram,
              "frequency" : frequencies,
              "time" : self.time}, orient = 'index')
        return df
    
    def EEMD(self,NE = None ,NSTD = None,) :
        if not NE :
            NE = 100
        if not NSTD :
            NSTD = 0.4
            
        def extrema(x,l) :
            local_max,local_min = [[0],[x[0]]],[[0],[x[0]]]
            for i in range(1,l - 1) :
                if x[i - 1] <= x[i] and x[i] >= x[i + 1] :
                    local_max[0].append(i)
                    local_max[1].append(x[i])
                if x[i - 1] >= x[i] and x[i] <= x[i + 1] :
                    local_min[0].append(i)
                    local_min[1].append(x[i])
            local_max[0].append(l - 1)
            local_max[1].append(x[l - 1])
            local_min[0].append(l - 1)
            local_min[1].append(x[l - 1])
            
            if len(local_max[0]) >= 4 :
                slope0 = (local_max[1][1] - local_max[1][2]) / (local_max[0][1] - local_max[0][2])
                ref = slope0 * (local_max[0][0] - local_max[0][1]) + local_max[1][1]
                if ref > local_max[1][0] :
                    local_max[1][0] = ref
                    
                slopen = (local_max[1][-2] - local_max[1][-3]) / (local_max[0][-2] - local_max[0][-3])
                ref = slopen * (local_max[0][-1] - local_max[0][-2]) + local_max[1][-2]
                if ref > local_max[1][-1] :
                    local_max[1][-1] = ref
                    
            if len(local_min[0]) >= 4 :
                slope0 = (local_min[1][1] - local_min[1][2]) / (local_min[0][1] - local_min[0][2])
                ref = slope0 * (local_min[0][0] - local_min[0][1]) + local_min[1][1]
                if ref < local_min[1][0] :
                    local_min[1][0] = ref
                
                slopen = (local_min[1][-2] - local_min[1][-3]) / (local_min[0][-2] - local_min[0][-3])
                ref = slopen * (local_min[0][-1] - local_min[0][-2]) + local_min[1][-2]
                if ref < local_min[1][-1] :
                    local_min[1][-1] = ref
            return np.array(local_max),np.array(local_min)
        
        def CreateEnvelope(x,N) :
            '''
            f = interp1d(x[0], x[1],kind = "cubic",fill_value = N)
            x_new = np.linspace(x[0][0],x[0][-1],N)
            return x_new,f(x_new)
            '''
            f = CubicSpline(x[0], x[1], bc_type='natural')
            x_new = np.linspace(x[0][0],x[0][-1],N)
            return x_new,f(x_new)
            

        x = self.data
        xsize,xSTD = self.length,np.std(self.data,ddof = 1)
        
        
        Nimf = int(np.floor(math.log2(xsize)) - 1)
        Nallmode = Nimf + 2
        allmode = np.zeros(shape = (Nallmode,xsize))
        
        for ensemble in range(NE) :
            #print("ensemble = ",ensemble)
            x = self.data / xSTD
            allmode[0] += x
            
            xorig = np.zeros(shape = (xsize,))
            
            for i in range(xsize) :
                xorig[i] = x[i] + np.random.normal(0,1,1) * NSTD
            
            xend = xorig
            for Nmode in range(1,Nimf + 1) :
                #print("Nmode",Nmode)
                xstart = xend
                for sift in range(10) :
                    #print("sift = ",sift)
                    local_max,local_min = extrema(xstart,xsize)
                    #print("nex = ",len(local_max[0]))
                    #if len(local_max[0]) <= 4 :
                        #break
                    __,MaxEnv = CreateEnvelope(local_max,xsize)
                    __,MinEnv = CreateEnvelope(local_min,xsize)
                    MeanEnv = (MaxEnv + MinEnv) / 2

                    xstart = xstart - MeanEnv
                imf = xstart
                allmode[Nmode] += imf
                xend = xend - imf

            allmode[-1] += xend
        allmode = allmode * xSTD / NE

        self.allmode = allmode
        '''
        for i in range(Nallmode) :
            plt.figure()
            plt.plot(self.time,allmode[i])
            if i == 0 :
                plt.title("original data with noise")
            else :
                plt.title('IMF' + str(i + 1))
        '''
        df = pd.DataFrame.from_dict({"allmode" : allmode}, orient = "index")
        return df
    
if __name__ == "__main__":
    '''
    fs = 256
    x = np.arange(0,2,1 / fs)
    intervals = np.array([[i if 0.2 < i < 0.6 else 0 for i in np.arange(0,2,1/fs)],
                 [i if 0.9 < i < 1.4 else 0 for i in np.arange(0,2,1/fs)]])
    freq = [20,80]
    data = np.zeros(shape = len(x))
    for i in range(len(freq)) :
        data += np.sin(2 * np.pi * freq[i] * intervals[i])
    '''
    fs = 1024
    t = np.arange(0, 1.0, 1.0 / fs)
    f1 = 100
    f2 = 200
    f3 = 300
    data = np.piecewise(t, [t < 1, t < 0.8, t < 0.3],
                    [lambda t: np.sin(2 * np.pi * f1 * t), 
                     lambda t: np.sin(2 * np.pi * f2 * t),
                     lambda t: np.sin(2 * np.pi * f3 * t)])
    
    #data = np.sin(2 * np.pi * 10 * x) + np.sin(2 * np.pi * 35 * x)
    test = time_frequency_analyze(data,fs)
    #test.FFT()
    #test.DWT("db4",3)
    #test.EEMD(100,0.4,1)
    test.CWT()

