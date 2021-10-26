import numpy as np
import scipy
import scipy.io.wavfile as wav
from scipy import signal
#from pylab import specgram
#import matplotlib.pyplot as plt
import os
#import cPickle as pickle
import h5py
import librosa

FRAMESIZE = 512
OVERLAP = 256
FFTSIZE = 512
RATE=16000
eps = np.finfo(float).eps
data_name = "/new_data/data_257_spectrum.h5"

noisylistpath = "/new_data/abnormal"
noisydata = np.zeros((30000000,257,5),dtype=np.float32)
idx = 0
idxx = 0
with open(noisylistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print idx
        y,sr=librosa.load(line[:-1],sr=16000)
        D=librosa.stft(y,n_fft=FRAMESIZE,hop_length=OVERLAP,win_length=FFTSIZE,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2) # Log-Power Spectrum
        print 'spec.shape' + str(Sxx.shape)
        mean = np.mean(Sxx, axis=1).reshape(257,1)
        std = np.std(Sxx, axis=1).reshape(257,1)
        Sxx = (Sxx-mean)/(std+eps) # normalization 
        for i in range(2, Sxx.shape[1]-2): # 5 Frmae
            idxx += 1
            if idxx%2==1: # Too many data, we skip half of them                
                continue        
            noisydata[idx,:,:] = Sxx[:,i-2:i+3]
            idx = idx + 1

noisydata = noisydata[:idx]
noisydata = np.reshape(noisydata,(idx,-1))

with h5py.File(data_name, 'a') as hf:
    hf.create_dataset('trnoisy', data=noisydata) # For Noisy data
noisdydata = []

cleanlistpath = "/new_data/normal"
cleandata = np.zeros((100000,257),dtype=np.float32)
c_idx = 0
idxx = 0
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print c_idx
        y,sr=librosa.load(line[:-1],sr=16000)
        D=librosa.stft(y,n_fft=FRAMESIZE,hop_length=OVERLAP,win_length=FFTSIZE,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2) 
        # Note: normalization is optional here
        print 'spec.shape' + str(Sxx.shape)
        for i in range(2, Sxx.shape[1]-2): # 5 Frmae                   
            idxx += 1
            if idxx%2==1:                
                continue 
            cleandata[c_idx,:] = Sxx[:,i]
            c_idx = c_idx + 1

cleandata = cleandata[:c_idx]
cleandata = np.tile(cleandata,(5*100,1)) # Note: Depends on how many noise types and SNRs

with h5py.File(data_name, 'a') as hf:
    hf.create_dataset('trclean', data=cleandata) # For Clean data
# =================================================================================
'''
noisylistpath = "/mnt/hd-01/user_sylar/MHINTSYPD_100NS/tsnoisylist"
noisydata = np.zeros((5000000,257,5),dtype=np.float32)
idx = 0
with open(noisylistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print idx
        y,sr=librosa.load(line[:-1],sr=16000)
        D=librosa.stft(y,n_fft=FRAMESIZE,hop_length=OVERLAP,win_length=FFTSIZE,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2) 
        print 'spec.shape' + str(Sxx.shape)
        maxxx = np.max(Sxx, axis=1).reshape(257,1)
        minxx = np.min(Sxx, axis=1).reshape(257,1)
        Sxx = 2*(Sxx-minxx)/(maxxx-minxx)-1 # [-1,1]  
        for i in range(2, Sxx.shape[1]-2): # 5 Frmae        
            noisydata[idx,:,:] = Sxx[:,i-2:i+3] # For Noisy data
            idx = idx + 1

noisydata = noisydata[:idx]
noisydata = np.reshape(noisydata,(idx,-1))

cleanlistpath = "/mnt/hd-01/user_sylar/MHINTSYPD_100NS/tscleanlist"
cleandata = np.zeros((100000,257),dtype=np.float32)
c_idx = 0
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print c_idx
        y,sr=librosa.load(line[:-1],sr=16000)
        D=librosa.stft(y,n_fft=FRAMESIZE,hop_length=OVERLAP,win_length=FFTSIZE,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2) 
        print 'spec.shape' + str(Sxx.shape)
        for i in range(2, Sxx.shape[1]-2): # 5 Frmae        
            cleandata[c_idx,:] = Sxx[:,i] # For Clean data
            c_idx = c_idx + 1

cleandata = cleandata[:c_idx]
cleandata = 2*(cleandata[:]-minx)/(maxx-minx)-1
cleandata = np.tile(cleandata,(2*2,1))

with h5py.File(data_name, 'a') as hf:
    hf.create_dataset('tsclean', data=cleandata) # For Clean data
    hf.create_dataset('tsnoisy', data=noisydata) # For Noisy data
'''

