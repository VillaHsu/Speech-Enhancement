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

eps = np.finfo(float).eps
def OverlapAdd(X, phase, windowLen=512, shift=256):
    FreqRes = X.shape[0]
    FrameNum = X.shape[1]
    Spec = X*np.exp(1j*phase)
    Spec = np.concatenate((Spec, np.flipud(np.conjugate(Spec[1:-1,:]))), axis=0)
    print Spec.shape
    sig = np.zeros((1,(FrameNum-1)*shift+windowLen))
    for i in range(0, FrameNum):
        start = i*shift
        s = Spec[:,i]
        sig[0,start:start+windowLen] = sig[0,start:start+windowLen] + np.real(np.fft.ifft(s, windowLen))
    return np.squeeze(sig)

def PowerSpectrum2Wave(log10powerspectrum,yphase):
    logpowspectrum = np.log(np.power(10, log10powerspectrum)) #log power spectrum
    sig            = OverlapAdd(np.sqrt(np.exp(logpowspectrum)), yphase)
    return sig

model=load_model("weights/DNN_spec_20160425v2.hdf5")
noisy_file = "test.wav"
y,sr=librosa.load(noisy_file,sr=16000)
training_phase = np.empty((1000, 257))   # For Noisy data
training_data = np.empty((1000, 257, 5)) # For Noisy data

D=librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
Sxx=np.log10(abs(D)**2)
phase=np.angle(D)*-1 #inverse phase sign  
mean = np.mean(Sxx, axis=1).reshape(257,1)
std = np.std(Sxx, axis=1).reshape(257,1)
Sxx = (Sxx-mean)/(std+eps) # Normalization   

idx = 0     
for i in range(2, Sxx.shape[1]-2): # 5 Frmae
    training_data[idx,:,:] = Sxx[:,i-2:i+3] 
    training_phase[idx,:] = phase[:,i]      
    idx = idx + 1
X_train = training_data[:idx]
X_train = np.reshape(X_train,(idx,-1))

phase = np.transpose(training_phase[:idx])
predict = model.predict(X_train)

p = np.transpose(predict)
out = PowerSpectrum2Wave(p, phase)
out = np.int16(out/np.max(np.abs(out)) * 32767)
wav.write("cleaned_"+noisy_file,16000,out)

#================================================================
# For multiple noisy files
'''
cleanlistpath = "/mnt/hd-01/user_sylar/MHINTSYPD_100NS/tscleanlist"
noisylistpath = "/mnt/hd-01/user_sylar/MHINTSYPD_100NS/tsnoisylist"
pathname = "20170508_DNN_spec"

with h5py.File("/mnt/hd-01/user_sylar/MHINTSYPD_100NS/data_257_spectrum.h5",'r') as hf:
    tr_min = np.array(hf.get("trmin"))
    tr_max = np.array(hf.get("trmax"))

noisy_list = []
with open(noisylistpath, 'r') as f:
    for line in f:
        noisy_list.append(line)
j=0
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = noisy_list[j].split('/')[-1][:-1]
        print filename
        y,sr=librosa.load(noisy_list[j][:-1],sr=16000)
        j+=1
        training_phase = np.empty((1000, 257))   # For Noisy data
        training_data = np.empty((1000, 257, 5)) # For Noisy data

        D=librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2)
        phase=np.angle(D)*-1 #inverse phase sign  
        maxx = np.max(Sxx, axis=1).reshape(257,1)
        minx = np.min(Sxx, axis=1).reshape(257,1)
        Sxx = 2*(Sxx-minx)/(maxx-minx)-1 # [-1,1]     

        idx = 0     
        for i in range(2, Sxx.shape[1]-2): # 5 Frmae
            training_data[idx,:,:] = Sxx[:,i-2:i+3] # For Noisy data
            training_phase[idx,:] = phase[:,i]      # For Noisy data
            idx = idx + 1

        X_train = training_data[:idx]
        X_train = np.reshape(X_train,(idx,-1))
        phase = np.transpose(training_phase[:idx])
        predict = model.predict(X_train)
        predict = (predict+1)/2*(tr_max-tr_min)+tr_min
        p = np.transpose(predict)
        out = PowerSpectrum2Wave(p, phase)
        out = np.int16(out*np.max(np.abs(out)) * 32767)
        wav.write("Enhanced/"+pathname+"/2girls/n3dB/"+filename,16000,out)
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = noisy_list[j].split('/')[-1][:-1]
        print filename
        y,sr=librosa.load(noisy_list[j][:-1],sr=16000)
        j+=1
        training_phase = np.empty((1000, 257))   # For Noisy data
        training_data = np.empty((1000, 257, 5)) # For Noisy data

        D=librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2)
        phase=np.angle(D)*-1 #inverse phase sign  
        maxx = np.max(Sxx, axis=1).reshape(257,1)
        minx = np.min(Sxx, axis=1).reshape(257,1)
        Sxx = 2*(Sxx-minx)/(maxx-minx)-1 # [-1,1]     

        idx = 0     
        for i in range(2, Sxx.shape[1]-2): # 5 Frmae
            training_data[idx,:,:] = Sxx[:,i-2:i+3] # For Noisy data
            training_phase[idx,:] = phase[:,i]      # For Noisy data
            idx = idx + 1

        X_train = training_data[:idx]
        X_train = np.reshape(X_train,(idx,-1))
        phase = np.transpose(training_phase[:idx])
        predict = model.predict(X_train)
        predict = (predict+1)/2*(tr_max-tr_min)+tr_min
        p = np.transpose(predict)
        out = PowerSpectrum2Wave(p, phase)
        out = np.int16(out*np.max(np.abs(out)) * 32767)
        wav.write("Enhanced/"+pathname+"/2girls/0dB/"+filename,16000,out)
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = noisy_list[j].split('/')[-1][:-1]
        print filename
        y,sr=librosa.load(noisy_list[j][:-1],sr=16000)
        j+=1
        training_phase = np.empty((1000, 257))   # For Noisy data
        training_data = np.empty((1000, 257, 5)) # For Noisy data

        D=librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2)
        phase=np.angle(D)*-1 #inverse phase sign  
        maxx = np.max(Sxx, axis=1).reshape(257,1)
        minx = np.min(Sxx, axis=1).reshape(257,1)
        Sxx = 2*(Sxx-minx)/(maxx-minx)-1 # [-1,1]     

        idx = 0     
        for i in range(2, Sxx.shape[1]-2): # 5 Frmae
            training_data[idx,:,:] = Sxx[:,i-2:i+3] # For Noisy data
            training_phase[idx,:] = phase[:,i]      # For Noisy data
            idx = idx + 1

        X_train = training_data[:idx]
        X_train = np.reshape(X_train,(idx,-1))
        phase = np.transpose(training_phase[:idx])
        predict = model.predict(X_train)
        predict = (predict+1)/2*(tr_max-tr_min)+tr_min
        p = np.transpose(predict)
        out = PowerSpectrum2Wave(p, phase)
        out = np.int16(out*np.max(np.abs(out)) * 32767)
        wav.write("Enhanced/"+pathname+"/2girls/n6dB/"+filename,16000,out)
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = noisy_list[j].split('/')[-1][:-1]
        print filename
        y,sr=librosa.load(noisy_list[j][:-1],sr=16000)
        j+=1
        training_phase = np.empty((1000, 257))   # For Noisy data
        training_data = np.empty((1000, 257, 5)) # For Noisy data

        D=librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2)
        phase=np.angle(D)*-1 #inverse phase sign  
        maxx = np.max(Sxx, axis=1).reshape(257,1)
        minx = np.min(Sxx, axis=1).reshape(257,1)
        Sxx = 2*(Sxx-minx)/(maxx-minx)-1 # [-1,1]     

        idx = 0     
        for i in range(2, Sxx.shape[1]-2): # 5 Frmae
            training_data[idx,:,:] = Sxx[:,i-2:i+3] # For Noisy data
            training_phase[idx,:] = phase[:,i]      # For Noisy data
            idx = idx + 1

        X_train = training_data[:idx]
        X_train = np.reshape(X_train,(idx,-1))
        phase = np.transpose(training_phase[:idx])
        predict = model.predict(X_train)
        predict = (predict+1)/2*(tr_max-tr_min)+tr_min
        p = np.transpose(predict)
        out = PowerSpectrum2Wave(p, phase)
        out = np.int16(out*np.max(np.abs(out)) * 32767)
        wav.write("Enhanced/"+pathname+"/crowd/n3dB/"+filename,16000,out)
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = noisy_list[j].split('/')[-1][:-1]
        print filename
        y,sr=librosa.load(noisy_list[j][:-1],sr=16000)
        j+=1
        training_phase = np.empty((1000, 257))   # For Noisy data
        training_data = np.empty((1000, 257, 5)) # For Noisy data

        D=librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2)
        phase=np.angle(D)*-1 #inverse phase sign  
        maxx = np.max(Sxx, axis=1).reshape(257,1)
        minx = np.min(Sxx, axis=1).reshape(257,1)
        Sxx = 2*(Sxx-minx)/(maxx-minx)-1 # [-1,1]     

        idx = 0     
        for i in range(2, Sxx.shape[1]-2): # 5 Frmae
            training_data[idx,:,:] = Sxx[:,i-2:i+3] # For Noisy data
            training_phase[idx,:] = phase[:,i]      # For Noisy data
            idx = idx + 1

        X_train = training_data[:idx]
        X_train = np.reshape(X_train,(idx,-1))
        phase = np.transpose(training_phase[:idx])
        predict = model.predict(X_train)
        predict = (predict+1)/2*(tr_max-tr_min)+tr_min
        p = np.transpose(predict)
        out = PowerSpectrum2Wave(p, phase)
        out = np.int16(out*np.max(np.abs(out)) * 32767)
        wav.write("Enhanced/"+pathname+"/crowd/0dB/"+filename,16000,out)
with open(cleanlistpath, 'r') as f:
    for line in f:
        filename = noisy_list[j].split('/')[-1][:-1]
        print filename
        y,sr=librosa.load(noisy_list[j][:-1],sr=16000)
        j+=1
        training_phase = np.empty((1000, 257))   # For Noisy data
        training_data = np.empty((1000, 257, 5)) # For Noisy data

        D=librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
        Sxx=np.log10(abs(D)**2)
        phase=np.angle(D)*-1 #inverse phase sign  
        maxx = np.max(Sxx, axis=1).reshape(257,1)
        minx = np.min(Sxx, axis=1).reshape(257,1)
        Sxx = 2*(Sxx-minx)/(maxx-minx)-1 # [-1,1]     

        idx = 0     
        for i in range(2, Sxx.shape[1]-2): # 5 Frmae
            training_data[idx,:,:] = Sxx[:,i-2:i+3] # For Noisy data
            training_phase[idx,:] = phase[:,i]      # For Noisy data
            idx = idx + 1

        X_train = training_data[:idx]
        X_train = np.reshape(X_train,(idx,-1))
        phase = np.transpose(training_phase[:idx])
        predict = model.predict(X_train)
        predict = (predict+1)/2*(tr_max-tr_min)+tr_min
        p = np.transpose(predict)
        out = PowerSpectrum2Wave(p, phase)
        out = np.int16(out*np.max(np.abs(out)) * 32767)
        wav.write("Enhanced/"+pathname+"/crowd/n6dB/"+filename,16000,out)
'''