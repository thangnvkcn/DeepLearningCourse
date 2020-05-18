import numpy as np


def _softmax(z):
    exp_vector = np.exp(-z)
    print(exp_vector)
    return exp_vector/np.sum(exp_vector, axis=0)

def _sigmoid(z):
    return 1/(1+np.exp(-z))

def rnn_cell_forward(xt, a_prev, parameters):
    Waa = parameters['Waa']
    ba = parameters['ba']
    Wya = parameters['Wya']
    by = parameters['by']
    Wax = parameters['Wax']
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    y = _softmax(np.dot(Wya, a_next) + by)
    cache = (a_next, a_prev, xt, parameters)

    return a_next, y, cache

def rnn_forward(x, a0, parameters):
    a_next = a0
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wya'].shape
    a = np.zeros(shape=(n_a, m, T_x))
    y = np.zeros(shape= (n_y, m, T_x))
    for t in range(T_x):
        a_next, y_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        caches.append(cache)
        a[:, :, t] = a_next
        y[:, :, t] = y_pred

    caches = (caches, x)
    return a, y, caches

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    Wf = parameters['Wf']
    bf = parameters['bf']
    Wu = parameters['Wu']
    bu = parameters['bu']
    Wc = parameters['Wc']
    bc = parameters['bc']
    Wo = parameters['Wo']
    bo = parameters['bo']
    Wy = parameters['Wy']
    by = parameters['by']

    concat = np.concatenate((a_prev, xt), axis=0)
    Lf = _sigmoid(np.dot(Wf, concat) + bf)
    Lu = _sigmoid(np.dot(Wu, concat) + bu)
    Lc = np.tanh(np.dot(Wc, concat) + bc)
    Lo = _sigmoid(np.dot(Wo, concat) + bo)
    c_next = Lf*c_prev + Lu*Lc
    a_next = np.tanh(c_next) * Lo
    y_pred = _softmax(np.dot(Wy, a_next) + by)

    cache = (a_next, c_next, a_prev, c_prev, Lf, Lu, Lc, Lo, xt, parameters)

    return a_next, c_next, y_pred, cache

def lstm_forward(x, a0, parameters):
    caches = []
    n_x, m, T_x = x.shape
    n_a, m = a0.shape
    n_y, n_a = parameters['Wy'].shape
    c = np.zeros(shape=(n_a, m, T_x))
    a = np.zeros(shape=(n_a, m, T_x))
    y = np.zeros(shape=(n_y, m, T_x))
    c_next = np.zeros_like(a0)
    a_next = a0
    for t in range(T_x):
        a_next, c_next, y_pred, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)
        c[:, :, t] = c_next
        a[:, :, t] = a_next
        y[:, :, t] = y_pred
        caches.append(cache)
    caches = (caches, x)
    return a, y, c, caches

def rnn_cell_backward(da_next, cache):
    (a_next, a_prev, xt, parameters) = cache
    dtanh = (1-a_next**2) * da_next
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    dxt = np.dot(Wax.T, dtanh) #34 - 42 = 32
    dWax = np.dot(dtanh, xt.T)
    dWaa = np.dot(dtanh, a_prev.T)
    da_prev = np.dot(Waa.T, dtanh)
    db = np.sum(dtanh, axis=1, keepdims=1)
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": db}
    return gradients

def rnn_backward(da, caches):
    print(da.shape)

    n_a, m, T_x = da.shape
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]
    n_x, m = x1.shape
    dx = np.zeros(shape=(n_x, m, T_x))
    dWax = np.zeros(shape=(n_a, n_x))
    dWaa = np.zeros_like(parameters['Waa'])
    dba = np.zeros_like(parameters['ba'])
    da0 = np.zeros_like(a0)
    da_prevt = np.zeros_like(a0)

    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:, :, t]+da_prevt, caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbt = gradients['dxt'], gradients['da_prev'], gradients['dWax'], gradients['dWaa'], gradients['dba']
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbt

    da0 = da_prevt
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
    return gradients

def lstm_cell_backward(da_next, dc_next, cache):
    (a_next, c_next, a_prev, c_prev, Lf, Lu, Lc, Lo, xt, parameters) = cache
    n_a, m = a_prev.shape
    n_x, m = xt.shape
    dot = da_next*np.tanh(c_next)*(Lo)*(1-Lo)
    dcct = da_next*Lo*(1-np.tanh(c_next)**2 + dc_next)*Lu*(1-Lc**2)
    dut = da_next*Lo*(1-np.tanh(c_next)**2 + dc_next)*Lc*Lu*(1-Lu)
    dft = da_next*Lo*(1-np.tanh(c_next)**2 + dc_next)*c_prev*Lf*(1-Lf)
    concat = np.concatenate((a_prev, xt), axis = 0)
    dWf = np.dot(dft, concat.T)
    dWu = np.dot(dut, concat.T)
    dWo = np.dot(dot, concat.T)
    dWc = np.dot(dcct, concat.T)
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbu = np.sum(dut, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)
    da_prev = np.dot(parameters['Wf'][:, :n_a].T,dft) + np.dot(parameters['Wo'][:, :n_a].T,dot) + np.dot(parameters['Wu'][:, :n_a].T,dut) + np.dot(parameters['Wc'][:, :n_a].T,dcct)
    dxt = np.dot(parameters['Wf'][:, n_a:].T,dft) + np.dot(parameters['Wo'][:, n_a:].T,dot) + np.dot(parameters['Wu'][:, n_a:].T,dut) + np.dot(parameters['Wc'][:, n_a:].T,dcct)
    dc_prev =da_next*Lo*(1-np.tanh(c_next)**2 + dc_next)*Lf
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWu": dWu, "dbu": dbu,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}
    return gradients

def lstm_backward(da, caches):
    (caches, x) = caches
    n_x, m, T_x = x.shape
    n_a, m, T_x = da.shape
    dc_next = np.zeros_like(da)
    (a_next, c_next, a_prev, c_prev, Lf, Lu, Lc, Lo, xt, parameters) = caches[0]
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a + n_x))
    dWu = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dbf = np.zeros((n_a, 1))
    dbu = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))

    for t in reversed(range(T_x)):
        gradients = lstm_cell_backward(da[:, :, t] + da_prevt,dc_prevt ,caches[t])
        dx[:, :, t] = gradients["dxt"]
        dWf += gradients["dWf"]
        dWu += gradients["dWu"]
        dWc += gradients["dWc"]
        dWo += gradients["dWo"]
        dbf += gradients["dbf"]
        dbu += gradients["dbu"]
        dbc += gradients["dbc"]
        dbo += gradients["dbo"]

    da0 = gradients["da_prev"]

    ### END CODE HERE ###

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWu": dWu, "dbu": dbu,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients
