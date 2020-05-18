import time
import dlib
import cv2
import keras.backend as K
from keras.applications import VGG16
from keras.layers import concatenate, Flatten, MaxPooling2D, Conv2D, Input, Lambda, Dropout,GlobalAveragePooling2D, Dense
# image = cv2.imread('face_reg.jpg')
from keras.models import Model
from keras.optimizers import SGD
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os
import random
from tqdm import tqdm

from sklearn.utils import shuffle
from keras.preprocessing.image import load_img, img_to_array
images = []
labels = []
names = []
count = 0

for path, dirs, files in os.walk('lfw-deepfunneled'):

    for d in tqdm(dirs):
#             print(count)

            list = os.listdir(os.path.join(path, d))  # dir is your directory path
            number_files = len(list)
            if(number_files > 1):
                added = 0
                for ext in ('jpg', 'jpeg', 'png'):
                    for f in glob.glob(os.path.join(path, d, '*.' + ext)):
                        img = load_img(f, target_size=(221, 221))
                        img = img_to_array(img)
                        if img is None:
                            continue
                        added = 1
                        images.append(img)
                        labels.append(count)
                if added == 1:
                    names.append(d)
                    count += 1
images, labels = shuffle(images, labels, random_state=0)



class Reg_Net():
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def Branch1(self):
        vgg_model = VGG16(weights=None, include_top = False, input_shape = self.input_shape)
        x = vgg_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Lambda(lambda x: K.l2_normalize(x, axis = 1))(x)
        conv_model = Model(inputs = vgg_model.input, outputs = x)
        return conv_model

    def Branch2(self):
        vgg_model = self.Branch1()
        inp1 = Input(shape=self.input_shape, name="input_branch2")
        conv1 = Conv2D(96, kernel_size=(8,8), strides=(16,16), padding='same')(inp1)
        max1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv1)
        max1 = Flatten()(max1)
        max1 = Lambda(lambda x: K.l2_normalize(x, axis = 1))(max1)

        inp2 = Input(shape=self.input_shape)
        conv2 = Conv2D(96, kernel_size=(8, 8), strides=(32, 32), padding='same')(inp2)
        max2 = MaxPooling2D(pool_size=(7, 7), strides=(4, 4), padding='same')(conv2)
        max2 = Flatten()(max2)
        max2 = Lambda(lambda x: K.l2_normalize(x, axis=1))(max2)

        merge1 = concatenate([max1, max2])
        merge2 = concatenate([merge1, vgg_model.output])
        emb = Dense(4096)(merge2)
        emb2 = Dense(128)(emb)
        l2_norm = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb2)

        final_model = Model(inputs = [vgg_model.input, inp1, inp2], outputs = l2_norm)
        return final_model


batch_size = 15
def loss_function(y_true, y_pred):
    loss = 0
    for i in range(0, batch_size//3, 3):
        q_pred = y_pred[i]
        p_pred = y_pred[i+1]
        n_pred = y_pred[i+2]
        alp = 1
        D_q_p = K.sqrt(K.sum(K.square(q_pred-p_pred)))
        D_q_n = K.sqrt(K.sum(K.square(q_pred-n_pred)))
        loss = loss + K.maximum(D_q_p-D_q_n+alp, 0)
    loss = loss/batch_size * 3
    return loss

net = Reg_Net((221, 221, 3))
model = net.Branch2()
model.compile(optimizer=SGD(lr=0.001,momentum=0.9, nesterov=True), loss=loss_function)

def image_batch_generator(images, labels, batch_size=24):
    lent_lab = len(labels)
    while True:
        batch_indice = np.random.choice(a = [i for i in range(len(labels))], size = batch_size//3)
        inp1 = []
        for i in range(len(batch_indice)):
            pos = np.where(labels == labels[i])[0]
            neg = np.where(labels != labels[i])[0]

            j = i
            while j==i:
                j = np.random.choice(pos)

            k = i
            while k==i:
                k = np.random.choice(neg)

            inp1.append(images[i])
            inp1.append(images[j])
            inp1.append(images[k])

        inp = np.array(inp1)
        inp1 = [inp, inp, inp]
        yield (inp1, np.zeros(batch_size))

import pickle
import matplotlib.pyplot as plt
X = np.array(images)
y = np.array(labels)

print(X.shape, " ", y.shape)

net = Reg_Net((221, 221, 3))
model = net.Branch2()

from keras.utils import plot_model

model.compile(optimizer=SGD(lr=0.001,momentum=0.9, nesterov=True), loss=loss_function)

model.fit_generator(generator=image_batch_generator(X, y, batch_size=15), steps_per_epoch=len(X)//batch_size, epochs=5)
model.save('model1.h5')

