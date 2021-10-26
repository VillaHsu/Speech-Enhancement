import numpy as np
import librosa
import random
import h5py

FRAMELENGTH = 10
OVERLAP = 10
path = "/mnt/hd-01/user_sylar/MHINTSYPD_100NS/"
data_name = "data_100_overlap.h5"

noisylistpath = path+"trnoisylist"
noisydata = np.zeros((300000000,FRAMELENGTH),dtype=np.float32)
idx = 0
with open(noisylistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print idx
        y,sr=librosa.load(line[:-1],sr=16000)
        for i in range(0, y.shape[0]-FRAMELENGTH, OVERLAP):
        	noisydata[idx] = y[i:i+FRAMELENGTH]
        	idx+=1
noisydata = noisydata[:idx]
noisydata = noisydata.reshape(idx,1,FRAMELENGTH,1)

cleanlistpath = path+"trcleanlist"
cleandata = np.zeros((2000000,FRAMELENGTH),dtype=np.float32)
c_idx = 0
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print c_idx
        y,sr=librosa.load(line[:-1],sr=16000)
        for i in range(0, y.shape[0]-FRAMELENGTH, OVERLAP):
        	cleandata[c_idx] = y[i:i+FRAMELENGTH]
        	c_idx+=1
cleandata = cleandata[:c_idx]
cleandata = np.tile(cleandata,(5*100,1)) # Note: Depends on how many noise types and SNRs

with h5py.File(path+data_name, 'a') as hf:
    hf.create_dataset('trclean', data=cleandata) # For Clean data
    hf.create_dataset('trnoisy', data=noisydata) # For Noisy data

# =================================================================================
'''
noisylistpath = path+"tsnoisylist"
noisydata = np.zeros((3000000,FRAMELENGTH),dtype=np.float32)
idx = 0
with open(noisylistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print filename
        y,sr=librosa.load(line[:-1],sr=16000)
        for i in range(0, y.shape[0]-FRAMELENGTH, OVERLAP):
        	noisydata[idx] = y[i:i+FRAMELENGTH]
        	idx+=1
noisydata = noisydata[:idx]
noisydata = noisydata.reshape(idx,1,FRAMELENGTH,1)

cleanlistpath = path+"tscleanlist"
cleandata = np.zeros((1000000,FRAMELENGTH),dtype=np.float32)
c_idx = 0
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print filename
        y,sr=librosa.load(line[:-1],sr=16000)
        for i in range(0, y.shape[0]-FRAMELENGTH, OVERLAP):
        	cleandata[c_idx] = y[i:i+FRAMELENGTH]
        	c_idx+=1
cleandata = cleandata[:c_idx]
cleandata = np.tile(cleandata,(2*2,1))

with h5py.File(path+"data_100_overlap.h5", 'a') as hf:
    hf.create_dataset('tsclean', data=cleandata) # For Clean data
    hf.create_dataset('tsnoisy', data=noisydata) # For Noisy data
'''
'''
noisydata = np.zeros((2000,512),dtype=np.float32)
y,sr=librosa.load(line[:-1],sr=16000)
# y = y/np.max(abs(y))
idx=0
for i in range(0, y.shape[0]-FRAMELENGTH, FRAMELENGTH):
    noisydata[idx] = y[i:i+FRAMELENGTH]
    idx+=1
noisydata = noisydata[:idx]
noisydata = noisydata.reshape(idx,1,512,1)
'''
