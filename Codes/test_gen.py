from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution2D, Convolution1D, MaxPooling2D, MaxPooling1D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adagrad, RMSprop
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.utils.io_utils import HDF5Matrix
from scipy import signal
import scipy.io
import scipy.io.wavfile as wav
import numpy as np
import h5py
import librosa

FRAMELENGTH = 512
OVERLAP = 128

model=load_model("weights/DNN_spec_20160425v2.hdf5")
noisy_file = "test.wav"
noisy_data = np.zeros((1000000,FRAMELENGTH))

y,sr=librosa.load(noisy_file,sr=16000)
# Zero padding
temp = np.zeros(((y.shape[0]//FRAMELENGTH+1)*FRAMELENGTH))
temp[:y.shape[0]] = y

# Slice wav to shape = (total_nums, 1, FRAMELENGTH, 1)
idx=0
for i in range(0, temp.shape[0]-FRAMELENGTH, OVERLAP):
    noisy_data[idx] = temp[i:i+FRAMELENGTH]
    idx+=1
noisy_data = noisy_data[:idx]
noisy_data = noisy_data.reshape(idx,1,FRAMELENGTH,1)
# Predict
enhanced = model.predict(noisy_data,verbose=1)
# Flatten enhanced segment to 1D waveform
idx=0
out = np.zeros(temp.shape[0])
for i in range(0, temp.shape[0]-FRAMELENGTH, OVERLAP):
    out[i:i+FRAMELENGTH] += enhanced[idx]
    idx+=1
out = out/(FRAMELENGTH//OVERLAP)
out = out[:y.shape[0]]
wav.write("cleaned_"+noisy_file,16000,np.int16(out*32767))

#=================================================================
# For multiple noisy files
'''
cleanlistpath = "/mnt/hd-01/user_sylar/MHINTSYPD_100NS/tscleanlist"
noisylistpath = "/mnt/hd-01/user_sylar/MHINTSYPD_100NS/tsnoisylist"
noisy_list = []
pathname = "20170518_FCN"
idx = 0
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print line[:-1]
        y,sr=librosa.load(line[:-1],sr=16000)
        out = np.zeros(y.shape[0])
        for i in range(0, y.shape[0]-FRAMELENGTH, OVERLAP):
            out[i:i+FRAMELENGTH] += enhanced[idx]
            idx+=1
        out = out/(FRAMELENGTH//OVERLAP)
        wav.write("Enhanced/"+pathname+"/2girls/n3dB/"+filename,16000,np.int16(out*32767))
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print line[:-1]
        y,sr=librosa.load(line[:-1],sr=16000)
        out = np.zeros(y.shape[0])
        for i in range(0, y.shape[0]-FRAMELENGTH, OVERLAP):
            out[i:i+FRAMELENGTH] += enhanced[idx]
            idx+=1
        out = out/(FRAMELENGTH//OVERLAP)
        wav.write("Enhanced/"+pathname+"/2girls/0dB/"+filename,16000,np.int16(out*32767))
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print line[:-1]
        y,sr=librosa.load(line[:-1],sr=16000)
        out = np.zeros(y.shape[0])
        for i in range(0, y.shape[0]-FRAMELENGTH, OVERLAP):
            out[i:i+FRAMELENGTH] += enhanced[idx]
            idx+=1
        out = out/(FRAMELENGTH//OVERLAP)
        wav.write("Enhanced/"+pathname+"/2girls/n6dB/"+filename,16000,np.int16(out*32767))
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print line[:-1]
        y,sr=librosa.load(line[:-1],sr=16000)
        out = np.zeros(y.shape[0])
        for i in range(0, y.shape[0]-FRAMELENGTH, OVERLAP):
            out[i:i+FRAMELENGTH] += enhanced[idx]
            idx+=1
        out = out/(FRAMELENGTH//OVERLAP)
        wav.write("Enhanced/"+pathname+"/crowd/n3dB/"+filename,16000,np.int16(out*32767))
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print line[:-1]
        y,sr=librosa.load(line[:-1],sr=16000)
        out = np.zeros(y.shape[0])
        for i in range(0, y.shape[0]-FRAMELENGTH, OVERLAP):
            out[i:i+FRAMELENGTH] += enhanced[idx]
            idx+=1
        out = out/(FRAMELENGTH//OVERLAP)
        wav.write("Enhanced/"+pathname+"/crowd/0dB/"+filename,16000,np.int16(out*32767))
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = line.split('/')[-1][:-1]
        print line[:-1]
        y,sr=librosa.load(line[:-1],sr=16000)
        out = np.zeros(y.shape[0])
        for i in range(0, y.shape[0]-FRAMELENGTH, OVERLAP):
            out[i:i+FRAMELENGTH] += enhanced[idx]
            idx+=1
        out = out/(FRAMELENGTH//OVERLAP)
        wav.write("Enhanced/"+pathname+"/crowd/n6dB/"+filename,16000,np.int16(out*32767))
'''
