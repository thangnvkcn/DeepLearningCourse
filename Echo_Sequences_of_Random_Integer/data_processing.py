import numpy as np
from pandas import  DataFrame, concat
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Conv1D

def generate_sequences(length):
    return np.random.randint(0, 100, (length,))


def one_hot_encode(sequence, n_unique = 100):
    return to_categorical(sequence, n_unique)

def one_hot_decode(encode_seq):
    return [np.argmax(x) for x in encode_seq]

def to_supervised(sequence, n_in, n_out):
    X = []
    y = []
    seq_len = len(sequence)
    sequence[-1]=0
    for i in range(seq_len):
        if(i+n_in > seq_len):
            break
        X.append(sequence[i: i+n_in])
        y.append(sequence[i: i+n_out])
        # print(X[i] , " =>>> ", y[i])
    return X, y

def get_data(n_in, n_out):
    sequence = generate_sequences(28)
    X, y = to_supervised(sequence, n_in, n_out)
    X_train, y_train = [], []
    for i in range(len(X)):
        X_train.append(one_hot_encode(X[i]))
        y_train.append(one_hot_encode(y[i]))
    return np.array(X_train), np.array(y_train)

def get_model(inp_shape):
    model = Sequential()
    model.add(LSTM(25, return_sequences=True, batch_input_shape=inp_shape, stateful=True))
    model.add(Conv1D(20, kernel_size=(2,)))
    model.add(TimeDistributed(Dense(100, activation='softmax')))
    return model

model = get_model((8, 5, 100))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# X, y = get_data(5, 5)
# print(X[0], " ##########################################", y[0])
for i in range(1000):
    X, y = get_data(5, 4)
    model.fit(X, y, batch_size=8, epochs=1, verbose=1, shuffle=False)
    model.reset_states()


X, y = get_data(5, 4)
yhat = model.predict(X, batch_size=8)

print(yhat.shape)

for i in range(len(yhat)):
    print("predicted = ", one_hot_decode(yhat[i]), " ######### actual = ", one_hot_decode(y[i]) )