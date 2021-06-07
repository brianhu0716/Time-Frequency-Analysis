# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 09:45:56 2021

@author: Brian Hu
"""

'''
import numpy as np
import pywt
import matplotlib.pyplot as plt 
from scipy.fftpack import fft
from scipy.signal import resample
import math
import random

class time_frequency_analyze():
    def __init__(self,data,fs) :
        self.data = data
        self.length = len(data)
        self.fs = fs
        self.time = np.arange(0,self.length / self.fs,1 / self.fs)
        
        plt.figure()
        plt.plot(self.time,self.data)
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.title("raw data")
        
    def Upsampling(self):
        if 2 ** (k := int(math.log2(self.length))) != self.length :  
            self.length = 2 ** (k + 1)
            self.data,self.time = resample(self.data,self.length,self.time)
            self.fs = self.length / (self.time[-1] - self.time[0])
        
    def FFT(self):
        n = self.length // 2
        f_axis = np.linspace(0,self.fs // 2,n)
        spec = fft(self.data)[: n]
        plt.figure()
        plt.plot(f_axis,abs(spec))
        plt.xlabel("frequency")
        plt.ylabel("energy")
        plt.title("FFT spectrum")
        
        # return spec
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
        #return coeffs
            
    #def CWT(self,beta,gamma) :
    
    def EEMD(self,NE = None ,NSTD = None) :
        x = self.data
        if not NE :
            NE = 100
        if not NSTD :
            NSTD = np.std(self.data)
        x = x / NSTD
        xsize = len(x);
        Nimf = np.floor(math.log2(xsize))-1;
        Nallmode = Nimf+2;
        # 初始化allmode以及mode矩陣
        allmode = np.zeros(shape = (Nallmode,xsize));
        mode = np.zeros(shape = (Nallmode,xsize))
        
        for n in range(NE) :
            xorig = x
            for i in range(xsize) :
                xorig[i] = xorig[i] + random.randint(0,1)
                
            mode[0] = xorig
            for i in range()

            
            
                
        
        
        
fs = 200
x = np.arange(0,2,1 / fs)
intervals = np.array([[i if 0.2 < i < 0.6 else 0 for i in np.arange(0,2,1/fs)],
             [i if 0.9 < i < 1.4 else 0 for i in np.arange(0,2,1/fs)]])
freq = [10,45]
data = np.zeros(shape = len(x))
for i in range(len(freq)) :
    data += np.sin(2 * np.pi * freq[i] * intervals[i])

#data = np.sin(2 * np.pi * 10 * x) + np.sin(2 * np.pi * 35 * x)
test = time_frequency_analyze(data,fs)
test.FFT()
test.DWT("db4",3)
'''

#ubuntu version
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 19:44:18 2021

@author: brian
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt 
from scipy.fftpack import fft
from scipy.signal import resample
import math
import random
from scipy.interpolate import interp1d

class time_frequency_analyze():
    def __init__(self,data,fs) :
        self.data = data
        self.length = len(data)
        self.fs = fs
        self.time = np.arange(0,self.length / self.fs,1 / self.fs)
        
        plt.figure()
        plt.plot(self.time,self.data)
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.title("raw data")
        
    def Upsampling(self):
        if 2 ** (k := int(math.log2(self.length))) != self.length :  
            self.length = 2 ** (k + 1)
            self.data,self.time = resample(self.data,self.length,self.time)
            self.fs = self.length / (self.time[-1] - self.time[0])
        
    def FFT(self):
        n = self.length // 2
        f_axis = np.linspace(0,self.fs // 2,n)
        spec = fft(self.data)[: n]
        plt.figure()
        plt.plot(f_axis,abs(spec))
        plt.xlabel("frequency")
        plt.ylabel("energy")
        plt.title("FFT spectrum")
        
        # return spec
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
        #return coeffs
            
    #def CWT(self,beta,gamma) :
    
    def EEMD(self,NE = None ,NSTD = None) :
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
            
            if local_max.shape(0) >= 4 :
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
            f = interp1d(x[0], x[1])
            x_new = np.linspace(x[0,0],x[0,-1],N)
            return x_new,f(x_new)
                    
                    
                    
        x = self.data
        if not NE :
            NE = 50
        if not NSTD :
            NSTD = np.std(self.data)
        x = x / NSTD
        xsize = len(x);
        Nimf = np.floor(math.log2(xsize))-1;
        Nallmode = Nimf + 2;
        # 初始化allmode以及mode矩陣
        allmode = np.zeros(shape = (Nallmode,xsize));
        mode = np.zeros(shape = (Nallmode,xsize))
        
        for _1 in range(NE) :
            xorig = x
            for i in range(xsize) :
                xorig[i] = xorig[i] + random.randint(0,1)
                
            mode[0] = x
            Nmode = 0
            xend = xorig
            while Nmode <= Nimf :
                Niter = 1
                xstart = xend
                while Niter <= 10:
                    local_max,local_min = extrema(xstart,xsize)
                    _,ymax = CreateEnvelope(local_max[0],local_max[1])
                    _,ymin = CreateEnvelope(local_min[0],local_min[1])
                    mean = (ymax + ymin) / 2
                    xstart -= mean
                    Niter += 1
                imf = xstart
                mode[Nmode + 1] += imf
                Nmode += 1
                xend -= imf
            mode[Nmode + 1] = xend
            allmode = allmode + mode
        allmode = allmode / NE
        allmode = allmode * NSTD
    
                
            
            
            
                
        
        
        
fs = 200
x = np.arange(0,2,1 / fs)
intervals = np.array([[i if 0.2 < i < 0.6 else 0 for i in np.arange(0,2,1/fs)],
             [i if 0.9 < i < 1.4 else 0 for i in np.arange(0,2,1/fs)]])
freq = [10,45]
data = np.zeros(shape = len(x))
for i in range(len(freq)) :
    data += np.sin(2 * np.pi * freq[i] * intervals[i])

#data = np.sin(2 * np.pi * 10 * x) + np.sin(2 * np.pi * 35 * x)
test = time_frequency_analyze(data,fs)
test.FFT()
test.DWT("db4",3)

