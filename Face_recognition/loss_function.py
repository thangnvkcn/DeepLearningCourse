import keras.backend as K
batch_size = 24


def loss_tensor(y_true, y_pred):
    loss = 0
    p = 1
    for i in range(0, batch_size, 3):
        q_emb = y_pred[i]
        p_emb = y_pred[i+1]
        n_emb = y_pred[i+2]
        Dq_p = K.sqrt(K.sum((q_emb - p_emb)**2))
        Dq_n = K.sqrt(K.sum((q_emb - n_emb)**2))
        loss += Dq_p-Dq_n+p

    loss = loss/batch_size * 3
    return max(Dq_p-Dq_n+p, 0)
