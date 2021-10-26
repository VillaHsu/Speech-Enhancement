# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:42:41 2016

@author: Jason
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, AtrousConv2D, ZeroPadding2D, UpSampling2D
from keras.layers.local import LocallyConnected2D
from keras.layers.merge import *
from keras.optimizers import *
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import *
import scipy.io
import time  
import numpy as np
import numpy.matlib
from model import *


date = "20160518"
data_path = "/mnt/hd-01/user_sylar/MHINTSYPD_100NS/data_512_nooverlap.h5"
start_time = time.time()
epoch=50
print 'model building...'
# UNET
#=========================================================
'''
inputs = Input((1,512,1))
conv1 = Conv2D(16, (11,1), padding='same', data_format="channels_first")(inputs)
conv1 = BatchNormalization(axis=1)(conv1)
conv1 = LeakyReLU(alpha=0.2)(conv1)
conv2 = Conv2D(32, (3,1), strides=(2,1), padding='same', data_format="channels_first")(conv1)
conv2 = BatchNormalization(axis=1)(conv2)
conv2 = LeakyReLU(alpha=0.2)(conv2)

conv2 = Conv2D(32, (3,1), padding='same', data_format="channels_first")(conv2)
conv2 = BatchNormalization(axis=1)(conv2)
conv2 = LeakyReLU(alpha=0.2)(conv2)
conv3 = Conv2D(64, (3,1), strides=(2,1), padding='same', data_format="channels_first")(conv2)
conv3 = BatchNormalization(axis=1)(conv3)
conv3 = LeakyReLU(alpha=0.2)(conv3)

conv3 = Conv2D(64, (3,1), padding='same', data_format="channels_first")(conv3)
conv3 = BatchNormalization(axis=1)(conv3)
conv3 = LeakyReLU(alpha=0.2)(conv3)
conv4 = Conv2D(128, (3,1), strides=(2,1), padding='same', data_format="channels_first")(conv3)
conv4 = BatchNormalization(axis=1)(conv4)
conv4 = LeakyReLU(alpha=0.2)(conv4)

conv4 = Conv2D(128, (3,1), padding='same', data_format="channels_first")(conv4)
conv4 = BatchNormalization(axis=1)(conv4)
conv4 = LeakyReLU(alpha=0.2)(conv4)
conv5 = Conv2D(256, (3,1), strides=(2,1), padding='same', data_format="channels_first")(conv4)
conv5 = BatchNormalization(axis=1)(conv5)
conv5 = LeakyReLU(alpha=0.2)(conv5)

conv5 = Conv2D(256, (3,1), padding='same', data_format="channels_first")(conv5)
conv5 = BatchNormalization(axis=1)(conv5)
conv5 = LeakyReLU(alpha=0.2)(conv5)
mid5 = Conv2D(512, (3,1), strides=(2,1), padding='same', data_format="channels_first")(conv5)
mid5 = BatchNormalization(axis=1)(mid5)
mid5 = LeakyReLU(alpha=0.2)(mid5)

mid5 = Conv2D(512, (3,1), activation='relu', padding='same', data_format="channels_first")(mid5)

up6 = concatenate([UpSampling2D(size=(2, 1),data_format="channels_first")(mid5), conv5], axis=1)
conv6 = Conv2D(256, (3,1), activation='relu', padding='same', data_format="channels_first")(up6)
conv6 = Conv2D(256, (3,1), activation='relu', padding='same', data_format="channels_first")(conv6)

up7 = concatenate([UpSampling2D(size=(2, 1),data_format="channels_first")(conv6), conv4], axis=1)
conv7 = Conv2D(128, (3,1), activation='relu', padding='same', data_format="channels_first")(up7)
conv7 = Conv2D(128, (3,1), activation='relu', padding='same', data_format="channels_first")(conv7)

up8 = concatenate([UpSampling2D(size=(2, 1),data_format="channels_first")(conv7), conv3], axis=1)
conv8 = Conv2D(64, (3,1), activation='relu', padding='same', data_format="channels_first")(up8)
conv8 = Conv2D(64, (3,1), activation='relu', padding='same', data_format="channels_first")(conv8)

up9 = concatenate([UpSampling2D(size=(2, 1),data_format="channels_first")(conv8), conv2], axis=1)
conv9 = Conv2D(32, (3,1), activation='relu', padding='same', data_format="channels_first")(up9)
conv9 = Conv2D(32, (3,1), activation='relu', padding='same', data_format="channels_first")(conv9)

up10 = concatenate([UpSampling2D(size=(2, 1),data_format="channels_first")(conv9), conv1], axis=1)
conv10 = Conv2D(16, (3,1), activation='relu', padding='same', data_format="channels_first")(up10)
conv10 = Conv2D(1, (1, 1), activation='linear')(conv10)
conv10 = Flatten()(conv10)

model = Model(inputs=inputs, outputs=conv10)
'''
# Fully-Convolution
#=========================================================
'''
model = Sequential()

model.add(Conv2D(32, (11, 1), padding='same', data_format="channels_first", input_shape=(1,512,1)))
model.add(BatchNormalization(axis=1))
model.add(ELU())
model.add(Dropout(0.07))

model.add(Conv2D(64, (11, 1), padding='same',  data_format="channels_first"))
model.add(BatchNormalization(axis=1))
model.add(ELU())
model.add(Dropout(0.07))

model.add(Conv2D(128, (11, 1), padding='same',  data_format="channels_first"))
model.add(BatchNormalization(axis=1))
model.add(ELU())
model.add(Dropout(0.07))

model.add(Conv2D(64, (11, 1), padding='same',  data_format="channels_first"))
model.add(BatchNormalization(axis=1))
model.add(ELU())
model.add(Dropout(0.07))

model.add(Conv2D(32, (11, 1), padding='same',  data_format="channels_first"))
model.add(BatchNormalization(axis=1))
model.add(ELU())
model.add(Dropout(0.07))

model.add(Conv2D(1, (11, 1), activation='linear', padding='same', data_format="channels_first"))
#model.add(Activation('tanh'))
model.add(Flatten())
'''
model = build_wav_generator(in_shp=(1,512,1), n_layers=3, ndf = 32)
model.summary()

sgd=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
rmsprop=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
adagrad=Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
adadelta=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
adamax=Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
nadam=Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
adam=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=adam,loss_weights=[1000])

print 'data loading...'
X_train = HDF5Matrix(data_path,"trnoisy",start=0,end=1000)
y_train = HDF5Matrix(data_path,"trclean",start=0,end=1000)
X_test = HDF5Matrix(data_path,"tsnoisy",start=0,end=1000)
y_test = HDF5Matrix(data_path,"tsclean",start=0,end=1000)


checkpointer = ModelCheckpoint(
						filepath="weights/FCN_"+date+".hdf5",
						monitor="loss",
						mode="min",
						verbose=0,
						save_best_only=True)
ES = EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=0, mode='auto')
TB = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
print 'training...'    
hist=model.fit(X_train, y_train, epochs=epoch, batch_size=200, verbose=1, shuffle="batch", callbacks=[checkpointer])



