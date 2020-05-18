from keras.layers import Dense
from keras.models import Sequential
import numpy as np
X = np.random.randint(0, 10, (50, 2))
y = 3*X[:, 0] + 2*X[:, 1]

print(X.shape, " ", y.shape)
model = Sequential()
model.add(Dense(1, input_shape=(2,)))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
for i in range(10000):
    model.fit(X, y, epochs=1)
    model.reset_states()
