import os
import pickle
from cv2 import imread
import numpy as np
from network_architacture import get_siamese_model
from keras.optimizers import Adam
from sklearn.utils import shuffle
import tensorflow as tf
from keras.models import load_model

def loadimgs(path, n=0):
    X = []
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n
    for alphabet in os.listdir(path):
        print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y, None]
        alphabet_path = os.path.join(path, alphabet)
        for letter in os.listdir(alphabet_path):
            cat_dict[letter] = (alphabet, letter)
            category_images = []
            letter_path = os.path.join(alphabet_path, letter)

            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = imread(image_path)
                category_images.append(image)
                y.append(curr_y)
            try:
                X.append(np.stack(category_images))
            except ValueError as e:
                print(e)
                print("error - category_images: ", category_images)
            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1
    y = np.vstack(y)
    X = np.stack(X)/255.
    # print(X)
    return (X, y, lang_dict)

# X_train, y_train, lang_dict1 = loadimgs(path='data/images_background')
# X_val, y_val, lang_dict2 = loadimgs(path='data/images_evaluation')
# print(X_train.shape)

X_train, X_val, y_train, y_val = np.random.randn(32, 20, 105, 105, 3), np.random.randn(1, 20, 105, 105, 3), np.random.randn(32, 1), np.random.randn(1, 1)
for i in range (3):
    with open('X_train' + str(i+1)+ '_triplet.pkl', 'rb') as f:
        # print(np.array(pickle.load(f)).shape)
        X_train =  np.concatenate((X_train,np.array(pickle.load(f))), axis=0)
        print(X_train.shape)
    with open('y_train' + str(i+1)+ '_triplet.pkl', 'rb') as f:
        # print(np.array(pickle.load(f)).shape)
        y_train =  np.concatenate((y_train,np.array(pickle.load(f))), axis=0)
    try:
        with open('X_val' + str(i+1)+ '_triplet.pkl', 'rb') as f:
            X_val =  np.concatenate((X_val,np.array(pickle.load(f))), axis=0)
        with open('y_val' + str(i+1)+ '_triplet.pkl', 'rb') as f:
            y_val =  np.concatenate((y_val,np.array(pickle.load(f))), axis=0)
    except:
        print("Xuat hien loi")
    print("Done "+str(i))

train_classes = 964
val_classes = 659
#
# #ghi du lieu vao file de train tren colab ko phai load lau
#
# with open('D:\Data\X_val1_triplet.pkl', 'wb') as f:
#     pickle.dump(X_val[:X_val.shape[0]//4], f)
# with open('D:\Data\y_val1_triplet.pkl', 'wb') as f:
#     pickle.dump(y_val[:X_val.shape[0]//4], f)
# with open('D:\Data\X_val2_triplet.pkl', 'wb') as f:
#     pickle.dump(X_val[X_val.shape[0]//4: X_val.shape[0]//2], f)
# with open('D:\Data\y_val2_triplet.pkl', 'wb') as f:
#     pickle.dump(y_val[X_val.shape[0]//4:X_val.shape[0]//2], f)
# with open('D:\Data\X_val3_triplet.pkl', 'wb') as f:
#     pickle.dump(X_val[X_val.shape[0]//2:X_val.shape[0]*3//4], f)
# with open('D:\Data\y_val3_triplet.pkl', 'wb') as f:
#     pickle.dump(y_val[X_val.shape[0]//2:X_val.shape[0]*3//4], f)
# with open('D:\Data\X_val4_triplet.pkl', 'wb') as f:
#     pickle.dump(X_val[X_val.shape[0]*3//4:], f)
# with open('D:\Data\y_val4_triplet.pkl', 'wb') as f:
#     pickle.dump(y_val[X_val.shape[0]*3//4:], f)




def get_batch(batch_size, s="train"):
    if s=="train":
        X = X_train
    else:
        X = X_val

    n_classes, n_examples, w, h, chanel = X.shape
    categories = np.random.choice(n_classes, size=(batch_size), replace=False)
    pairs = [np.zeros((batch_size,  h, w, 3)) for i in range(2)]
    # pairs = np.array(pairs)
    targets = np.zeros((batch_size,))
    targets[batch_size//2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = np.random.randint(0, n_examples)
        pairs[0][i, :, :, :] = X[category, idx_1].reshape(w,h,3)

        idx_2 = np.random.randint(0, n_examples)
        if i<batch_size//2:
            category2 = (category + np.random.randint(1, n_classes))%n_classes
        else:
            category2 = category
        pairs[1][i, :, :, :] = X[category2, idx_2].reshape(w,h, 3)
    pairs[0], pairs[1], targets = shuffle(pairs[0], pairs[1], targets)
    return pairs, targets

def generate(batch_size, s="train"):
    """
    a generator for batches, so model.fit_generator can be used.
    """
    while True:
        pairs, targets = get_batch(batch_size,s)
        yield (pairs, targets)

def make_oneshot_task(N, s="val", language=None):
    if s=="train":
        X = X_train
    else:
        X = X_val

    n_classes, n_examples, w, h, chanel = X.shape
    test = np.random.choice(n_classes, size=(1), replace=False)[0]
    targets = np.zeros((N,))
    targets[N-1] = 1
    idx_1 = np.random.randint(0, n_examples)
    id_1 = idx_1
    input1 = X[test][idx_1]
    input2 = []

    while id_1 == idx_1:
        id_1 = np.random.randint(0, n_examples)

    for i in range(N-1):
        test2 = (test + np.random.choice([i for i in range(1, n_classes)], size=(1), replace=False)[0]) % n_classes
        idx_2 = np.random.randint(0, n_examples)
        input2.append(X[test2][idx_2])

    input2.append(X[test][id_1])
    input1 = [input1 for i in range(N)]
    input1, input2, targets = shuffle(input1, input2, targets)
    return [input1, input2], targets

def test_oneshot(model, N, k, s="val", verbose=0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k, N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N, s)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct += 1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct, N))
    return percent_correct

# model = get_siamese_model(input_shape=(105, 105, 3))
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if test_oneshot(model, 4, 10) >= 90:
            print("\n stopping training!!")
            self.model.stop_training = True

callbacks = myCallback()
print("done")

model = load_model('model.h5')
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00006), metrics=['accuracy'])
model.fit_generator(generator=generate(32), steps_per_epoch=train_classes//32, epochs=100, callbacks = [callbacks])
model.save('model.h5')