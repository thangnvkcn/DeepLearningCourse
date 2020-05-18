from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import cv2
from matplotlib import pyplot as plt
import glob
import os
import random
import dlib
from tqdm import tqdm

def read_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (221, 221))

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_face_detector = dlib.get_frontal_face_detector()
    faces_hog = hog_face_detector(img, 1)

    if len(faces_hog) == 1:
        x = faces_hog[0].left()
        y = faces_hog[0].top()
        w = faces_hog[0].right() - x
        h = faces_hog[0].bottom() - y

        return img[y:y + h, x:x + w]

    #     print(image_path, len(faces_hog))

    return None

from sklearn.utils import shuffle
from keras.preprocessing.image import load_img, img_to_array
images = []
labels = []
names = []
count = 0

for path, dirs, files in os.walk('Image'):

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


import numpy as np
# model = load_model('model (2).h5', compile=False)
# model2 = load_model('model.h5', compile=False)

img1 = load_img('Image/Adam_Mair/Adam_Mair_0001.jpg', target_size=(221, 221))
img1 = img_to_array(img1)
img1 = np.expand_dims(img1, axis= 0)
images = np.array(images)

import numpy as np
imglb1 = model.predict([img1, img1, img1])
imglb2 = model2.predict([img1, img1, img1])
leng = images.shape[0]
db = model.predict([images, images, images])

minimum = 999999999
lb1 = np.linalg.norm(imglb1, axis=1)
lb2 = np.linalg.norm(imglb2, axis=1)

print(imglb1)
print(imglb2)

