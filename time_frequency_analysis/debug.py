#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 02:36:20 2021

@author: brian
"""
import numpy as np
import matplotlib.pyplot as plt 
import random
from scipy.interpolate import splev,splrep,interp1d,CubicSpline

def CreateEnvelope(x,N) :
    '''
    model = splrep(x[0],x[1],k=3) #生成模型参数
    x_new = np.linspace(0,1.995,N)
    y_new = splev(x_new,model) #生成插值点
    return x_new,y_new
    '''
    '''
    f = interp1d(x[0], x[1],kind = "cubic",fill_value = N)
    x_new = np.linspace(x[0][0],x[0][-1],N)
    return x_new,f(x_new)
    '''
    f = CubicSpline(x[0], x[1],bc_type = "natural")
    x_new = np.linspace(x[0][0],x[0][-1],N)
    return x_new,f(x_new)
    
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

for _ in range(6) :
    fs = 200
    x = np.arange(0,2,1 / fs)
    intervals = np.array([[i if 0.2 < i < 0.6 else 0 for i in np.arange(0,2,1/fs)],
                 [i if 0.9 < i < 1.4 else 0 for i in np.arange(0,2,1/fs)]])
    freq = [random.randint(1,80),random.randint(1,80)]
    data = np.zeros(shape = len(x))
    for i in range(len(freq)) :
        data += np.sin(2 * np.pi * freq[i] * intervals[i])
    
    data = data / np.std(data,ddof = 1)
    
    for i in range(len(data)) :
        data[i] += np.random.normal(0,1,1) * 0.4
    
    local_max,local_min = extrema(data,len(data))
    
    _,ymax = CreateEnvelope(local_max,len(data))
    _,ymin = CreateEnvelope(local_min,len(data))
    
    plt.figure()
    plt.plot(x,data,label = "orig")
    plt.plot(local_max[0] / fs,local_max[1],'o',label = "local max")
    plt.plot(local_min[0] / fs,local_min[1],'o',label = "local min")
    plt.plot(x,(ymax + ymin) / 2,label = "mean Envelop",linestyle = "--")
    #plt.plot(x,ymax,label = "max envelop",linestyle = "--")
    #plt.plot(x,ymin,label = "min envelop",linestyle = "--")
    plt.legend(loc = "best")
