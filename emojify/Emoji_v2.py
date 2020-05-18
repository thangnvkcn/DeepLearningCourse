import numpy as np
import emoji
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

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

def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros(shape=(m, max_len))
    for i in range(m):
        words = [word.lower() for word in X[i].split()]

        for j in range(len(words)):
            X_indices[i, j] = word_to_index[words[j]]
    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map['cucumber'].shape[0]
    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False, input_length=10)
    embedding_layer.build((-2,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer


def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(5)(X)
    X = Activation('softmax')(X)
    model = Model(inputs = sentence_indices, outputs = X)
    return model

maxLen = 10
model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
X_train_indices = sentences_to_indices(X_train, word_to_index, max_len=10)
model.fit(X_train_indices, Y_train, batch_size=32, epochs=50, shuffle=True)
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=10)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != np.argmax(Y_test[i]) ):
        print('Expected emoji: #'+ str(np.argmax(Y_test[i])) + ' prediction: '+ X_test[i] +" #"+ str(num))

x_test = np.array(['not feeling happy'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  str(np.argmax(model.predict(X_test_indices))))