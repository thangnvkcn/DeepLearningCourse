import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
filename = 'wonderland.txt'
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = {ch:i for i, ch in enumerate(chars)}
print(char_to_int)
n_chars = len(raw_text)
n_vocab = len(chars)
print(n_chars, " ", n_vocab)

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i: i+seq_length]
    seq_out = raw_text[i+seq_length]
    dataX.append([ char_to_int[ch] for ch in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)

X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X/float(n_vocab)
y = np_utils.to_categorical(dataY)

def model_and_train():
    model = Sequential([LSTM(256, input_shape=(X.shape[1], X.shape[2])),
                        Dropout(0.2),
                        Dense(y.shape[1], activation='softmax')])
    # model = Sequential()
    # model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    # model.add(Dropout(0.2))
    # model.add(Dense(y.shape[1], activation='softmax'))

    print("ok")
    model.compile(loss='categorical_crossentropy', optimizer='adam')


    filepath = "model-{loss:.4f}-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callback_list = [checkpoint]
    model.fit(X, y, epochs=20, batch_size=128, callbacks=callback_list)

model = load_model('model.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = {i:c for i, c in enumerate(chars)}
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
seq_in = [int_to_char[value] for value in pattern]
print("input: ", ''.join(seq_in))
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x/float(n_vocab)
    pred = model.predict(x, verbose = 0)
    index = np.argmax(pred)
    result = int_to_char[index]
    seq_in.append(result)
    pattern.append(index)
    pattern = pattern[1:]

print("text generation: ")
print(''.join(seq_in))