from keras.models import load_model, Model
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.activations import *
from keras.layers.convolutional import Conv2D, Conv1D, ZeroPadding2D, UpSampling2D, Cropping2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.normalization import *
from keras.layers.merge import *
from keras.initializers import *
from keras.utils import plot_model
# Build Generative model ...
# Generator
conv_init = RandomNormal(mean=0.0, stddev=0.02, seed=0)
def conv_batch_lrelu(conv, nf, kw, kh, sw, sh):
    conv1 = Conv2D(nf, (kw,kh), strides=(sw,sh), padding='same', kernel_initializer=conv_init, data_format="channels_first")(conv)
    # Note: Use BatchNormalization is optional
    # conv1 = BatchNormalization(axis=1)(conv1)
    # conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = PReLU(shared_axes=[1, 2])(conv1)
    return conv1

def conv_upsample_relu(conv, encoder_conv, nf, kw, kh, sw, sh):
    up = concatenate([UpSampling2D(size=(sw,sh),data_format="channels_first")(conv), encoder_conv], axis=1)
    conv1 = Conv2D(nf, (kw,kh), activation='relu', padding='same', data_format="channels_first")(up)
    conv1 = Conv2D(nf, (kh,kw), activation='relu', padding='same', data_format="channels_first")(conv1)
    return conv1

def build_wav_generator(in_shp=(1,16384,1), n_layers=2, ndf=16):

    kw,kh = 3,1
    sw,sh = 2,1
    encoder = []
    inputs = Input(in_shp)
    encoder.append(conv_batch_lrelu(inputs, ndf, kw, kh, 1, 1))
    for i in range(n_layers+1):
        x = conv_batch_lrelu(encoder[-1], ndf*min(2**(i+1),8), kw, kh, sw, sh)
        encoder.append(conv_batch_lrelu(x, ndf*min(2**(i+1),8), kw, kh, 1, 1))
    decoder = []
    decoder.append(conv_upsample_relu(encoder[-1], encoder[-2], ndf*min(2**n_layers,8), kw, kh, sw, sh))
    for i in range(n_layers):
        decoder.append(conv_upsample_relu(decoder[-1], encoder[-1*(i+3)], ndf*min(2**(n_layers-i-1),8), kw, kh, sw, sh))

    conv_last = Conv2D(1, (1, 1), activation='linear',data_format="channels_first")(decoder[-1])
    # conv10 = add([inputs, conv_last])
    conv10 = Flatten()(conv_last)
    generator = Model(inputs=inputs, outputs=conv10)
    plot_model(generator, show_shapes=True, to_file='generator.png')
    return generator

def build_wav_generator_multiply(in_shp=(1,16384,1), n_layers=2, ndf=16):

    kw,kh = 3,1
    sw,sh = 2,1
    encoder = []
    inputs = Input(in_shp)
    encoder.append(conv_batch_lrelu(inputs, ndf, kw*min(2**n_layers,4), kh, 1, 1))
    for i in range(n_layers+1):
        x = conv_batch_lrelu(encoder[-1], ndf*min(2**(i+1),8), kw*min(2**(n_layers-i),4), kh, sw, sh)
        encoder.append(conv_batch_lrelu(x, ndf*min(2**(i+1),8), kw*min(2**(n_layers-i),4), kh, 1, 1))
    decoder = []
    decoder.append(conv_upsample_relu(encoder[-1], encoder[-2], ndf*min(2**n_layers,8), kw, kh, sw, sh))
    for i in range(n_layers):
        decoder.append(conv_upsample_relu(decoder[-1], encoder[-1*(i+3)], ndf*min(2**(n_layers-i-1),8), kw, kh, sw, sh))

    conv_last = Conv2D(1, (1, 1), activation='linear')(decoder[-1])
    conv10 = multiply([inputs, conv_last])
    conv10 = Flatten()(conv10)
    generator = Model(inputs=inputs, outputs=conv10)
    plot_model(generator, show_shapes=True, to_file='generator.png')
    return generator

