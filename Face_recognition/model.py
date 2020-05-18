from keras.applications import VGG16
from keras.layers import concatenate, Flatten, MaxPooling2D, Conv2D, Input, GlobalAveragePooling2D, Dense, Dropout, Lambda
import keras.backend as K
from keras.models import Model
import tensorflow as tf
import numpy as np


class train_net:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def ConvNet_model(self):
        vgg_model = VGG16(weights=None, include_top=False, input_shape=self.input_shape)
        x = vgg_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Lambda(lambda x_: K.l2_normalize(x_, axis=1))(x)
        convnet_model = Model(inputs=vgg_model.input, outputs=x)
        return convnet_model

    def deep_rank_model(self):
        convnet_model = self.ConvNet_model()

        first_inp = Input(shape=self.input_shape)
        first_conv = Conv2D(96, kernel_size=(8, 8), strides=(16, 16), padding='same')(first_inp)
        first_max = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(first_conv)
        first_max = Flatten()(first_max)
        first_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(first_max)

        second_inp = Input(shape=self.input_shape)
        second_conv = Conv2D(96, kernel_size=(8, 8), strides=(32, 32), padding='same')(second_inp)
        second_max = MaxPooling2D(pool_size=(7, 7), strides=(4, 4), padding='same')(second_conv)
        second_max = Flatten()(second_max)
        second_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(second_max)
        print(first_max.shape)
        print(second_max.shape)
        merge_one = concatenate([first_max, second_max])
        merge_two = concatenate([merge_one, convnet_model.output])
        emb = Dense(4096)(merge_two)
        emb = Dense(128)(emb)
        l2_normal_final = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)

        final_model = Model(inputs = [first_inp, second_inp, convnet_model.input], outputs= l2_normal_final)
        return final_model

# net = train_net((221, 221, 3))
# print(net.deep_rank_model().output)

