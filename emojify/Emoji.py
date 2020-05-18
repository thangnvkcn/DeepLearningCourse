import numpy as np
import emoji
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.utils import to_categorical

def read_file(path):
    data = read_csv(path, header=None).to_numpy()
    X = data[:, 0]
    Y = data[:, 1]
    return X, Y


def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}

        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    word_to_index = {word:i for i, word in enumerate(words)}
    index_to_word = {i:word for i, word in enumerate(words)}
    return word_to_index, index_to_word, word_to_vec_map

X_train, Y_train = read_file('emojify/train_emoji.csv')
X_test, Y_test = read_file('emojify/test_emoji.csv')
maxlen = len(max(X_train).split())
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

def sentence_to_avg(sentence, word_to_vec_map):
    lower_words = sentence.lower().split()

    m = [word_to_vec_map[word] for word in lower_words]
    avg = np.average(m, 0)
    return avg

def _softmax(z):
    e = np.exp(z - np.max(z))
    return e/np.sum(e, axis=0)

def predict(X, Y, W, b, word_to_vec_map):
    m = Y.shape[0]
    y = []
    count = 0
    for i in range(m):
        Xi = sentence_to_avg(X[i], word_to_vec_map)
        y_pred = _softmax(np.dot(W, Xi) + b)
        y.append(np.argmax(y_pred))
        if np.argmax(y_pred) == np.argmax(Y[i]):
            count+=1
    print("Accuracy: ", count/len(X))
    return y

def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400 ):
    m = Y.shape[0]
    n_y = 5
    n_h = 50
    W = np.random.randn(n_y, n_h)/np.sqrt(n_h)
    b = np.random.rand(1,)
    for t in range(num_iterations):
        dW = 0
        db = 0
        for i in range(m):
            Xi = sentence_to_avg(X[i], word_to_vec_map)
            y_pred = _softmax(np.dot(W, Xi) + b)
            cost = -np.log(y_pred[np.argmax(Y[i])])
            dz = y_pred - Y[i]
            dW += np.dot(dz.reshape(-1, 1), Xi.reshape(-1, 1).T)
            db += np.sum(dz, axis = 0)

        W -=learning_rate* dW
        b -=learning_rate* db

        if t%100==0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b





Y = Y_train
X = X_train
print(Y.shape)
_, W, b = model(X, Y, word_to_vec_map)

X_my_sentences = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])
Y_my_labels = to_categorical(Y_my_labels, 5)

y_pred = predict(X_my_sentences, Y_my_labels, W, b, word_to_vec_map)
print(y_pred)