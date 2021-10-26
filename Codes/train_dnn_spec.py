# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:42:41 2016

@author: Jason
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, AtrousConv2D, ZeroPadding2D
from keras.layers.local import LocallyConnected2D
from keras.optimizers import *
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import *
import time  
import numpy as np
import numpy.matlib
import h5py
import scipy.io
from scipy import signal
import scipy.io.wavfile as wav
import librosa


date = "20160508"
data_path = "/mnt/hd-01/user_sylar/MHINTSYPD_100NS/data_257_spectrum.h5"
start_time = time.time()
epoch=30
print 'model building...'
model = Sequential()

# model.add(Reshape((2048,), input_shape=(1,2048,1)))

model.add(Dense(2048, input_shape=(1285,)))
model.add(BatchNormalization(axis=1))
model.add(ELU())
# model.add(Dropout(0.05))

model.add(Dense(2048))
model.add(BatchNormalization(axis=1))
model.add(ELU())
# model.add(Dropout(0.05))

model.add(Dense(2048))
model.add(BatchNormalization(axis=1))
model.add(ELU())
# model.add(Dropout(0.05))

model.add(Dense(2048))
model.add(BatchNormalization(axis=1))
model.add(ELU())
# model.add(Dropout(0.05))

model.add(Dense(2048))
model.add(BatchNormalization(axis=1))
model.add(ELU())
# model.add(Dropout(0.05))

model.add(Dense(257))
#model.add(Activation('tanh'))
model.summary()


adam=Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=adam, loss_weights=[1000])

print 'data loading...'
X_train = HDF5Matrix(data_path,"trnoisy")
y_train = HDF5Matrix(data_path,"trclean")
X_test = HDF5Matrix(data_path,"tsnoisy")
y_test = HDF5Matrix(data_path,"tsclean")


checkpointer = ModelCheckpoint(
						filepath="weights/DNN_spec_"+date+".hdf5",
						monitor="loss",
						mode="min",
						verbose=0,
						save_best_only=True)
ES = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
print 'training...'    
hist=model.fit(X_train, y_train, epochs=epoch, batch_size=200, verbose=1,shuffle="batch", callbacks=[checkpointer])

'''
# enhanced = scipy.io.loadmat('TrainTest_100_noise_types/Enhanced_wavfeorm.mat')['out'].astype(np.float32)
FRAMELENGTH = 2048
cleanlistpath = "/mnt/hd-01/user_sylar/TrainTest_100_noise_types/tscleanlist"
pathname = "20170418_DNN"
idx=0
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print line[:-1]
        y,sr=librosa.load(line[:-1],sr=16000)
        out = []
        for i in range(0, y.shape[0]-FRAMELENGTH, FRAMELENGTH):
            out.append(enhanced[idx])
            idx+=1
        out = np.reshape(out,-1)
        wav.write("Enhanced/"+pathname+"/crowd/0dB/"+filename,16000,np.int16(out*32767))
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print line[:-1]
        y,sr=librosa.load(line[:-1],sr=16000)
        out = []
        for i in range(0, y.shape[0]-FRAMELENGTH, FRAMELENGTH):
            out.append(enhanced[idx])
            idx+=1
        out = np.reshape(out,-1)
        wav.write("Enhanced/"+pathname+"/crowd/5dB/"+filename,16000,np.int16(out*32767))
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print line[:-1]
        y,sr=librosa.load(line[:-1],sr=16000)
        out = []
        for i in range(0, y.shape[0]-FRAMELENGTH, FRAMELENGTH):
            out.append(enhanced[idx])
            idx+=1
        out = np.reshape(out,-1)
        wav.write("Enhanced/"+pathname+"/2girls/0dB/"+filename,16000,np.int16(out*32767))
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print line[:-1]
        y,sr=librosa.load(line[:-1],sr=16000)
        out = []
        for i in range(0, y.shape[0]-FRAMELENGTH, FRAMELENGTH):
            out.append(enhanced[idx])
            idx+=1
        out = np.reshape(out,-1)
        wav.write("Enhanced/"+pathname+"/2girls/5dB/"+filename,16000,np.int16(out*32767))

'''
