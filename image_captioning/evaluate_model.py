from keras.applications import VGG16
from keras.models import Model, load_model
from os import  listdir
from keras_preprocessing.image import load_img, img_to_array
import pickle
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import  to_categorical

def extract_features(directory):
    model = VGG16(weights = 'imagenet')
    model = Model(inputs = model.input, outputs = model.layers[-2].output)
    model.summary()
    features = dict()
    for name in listdir(directory):
        filename = directory+ '/'+name
        image = load_img(filename, target_size=(224,224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        feature = model.predict(image, verbose=0)
        features[name.split('.')[0]] = np.reshape(feature, (4096,))
    return features

def load_doc(filename):
    file = open(filename, 'r', encoding='utf8')
    text = file.read()
    file.close()
    return text

def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line)<2:
            continue
        image_id, image_des = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_des = ' '.join(image_des)
        if image_id not  in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_des)
    return mapping

import string

def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, list_des in descriptions.items():
        for i in range(len(list_des)):
            des = list_des[i]
            des = des.lower()
            des = [w.translate(table) for w in des.split()]
            des = [w for w in des if len(w) > 1]
            desc = [w for w in des if w.isalpha()]
            list_des[i] = ' '.join(des)

def to_vocabulary(descriptions):
    vocab = set()
    vocab_count = dict()
    for key, desc_file in descriptions.items():
        for i in range(len(desc_file)):
            vocab.update(desc_file[i].split())
            for k in desc_file[i].split():
                if k not in vocab_count:
                    vocab_count[k] = 0
                vocab_count[k] += 1
    return vocab



def save_descriptions(descriptions, filename):
    mapping = list()
    with open(filename, 'w') as f:
        for key, desc_file in descriptions.items():
            for des in desc_file:
                mapping.append(key + ' ' + des)
        f.write('\n'.join(mapping))


def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return dataset

def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_des = tokens[0], tokens[1:]
        if image_id in dataset:
            image_des = ' '.join(image_des)
            if image_id not in mapping:
                mapping[image_id] = list()
            mapping[image_id].append('startseq ' + image_des + ' endseq')
    return mapping

def load_photo_features(filename, dataset):
    all_features = pickle.load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features


def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        for d in descriptions[key]:
            all_desc.append(d)
    return all_desc

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    # print(lines)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def maxlength(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


def data_generator(tokenizer, descriptions, photos, max_length, vocab_size):
    while True:
        for key, list_desc in descriptions.items():
            photo = photos[key]
            X1, X2, y = create_sequences(tokenizer, max_length, list_desc, photo, vocab_size)
            yield [[X1, X2], y]

from keras.models import Model
from keras.layers import Dense, LSTM, Input, Concatenate, Dropout, Embedding, add
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

def get_model(vocab_size, max_length):
    inp1 = Input(shape=(4096,))
    dr1 = Dropout(0.5)(inp1)
    img_ex = Dense(256, activation='relu')(dr1)

    inp2 = Input(shape=(max_length,))
    dr2 = Dropout(0.5)(inp2)
    emb = Embedding(vocab_size, 256, mask_zero=True)(dr2)
    seq_ex = LSTM(256)(emb)

    decoder1 = add([img_ex, seq_ex])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outp = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs = [inp1, inp2], outputs = outp)
    model.compile(loss='categorical_crossentropy', optimizer='adam' )
    plot_model(model, 'model.png', show_shapes=True)
    return model

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            print("idx, inte = ", index, " ", integer)
            return word
    return None

def genarate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        print(in_text)
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        yhat = model.predict([photo,seq])
        decode = np.argmax(yhat)
        word = word_for_id(decode, tokenizer)
        print(word)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text

filename = 'D:\Data\\New folder\Flickr8k_text\Flickr_8k.trainImages.txt'
#load set of filenames
train = load_set(filename)
#get dictionary include filename and descriptions about image
train_descriptions = load_clean_descriptions('descriptions.txt', train)

#get dictionary include filename and features of images
train_features = load_photo_features('features.pkl', train)

#get tokenizer on traing set
tokenizer = create_tokenizer(train_descriptions)
#get vocab size
vocab_size = len(tokenizer.word_index) + 1
# print(vocab_size)
maxlen = maxlength(train_descriptions)

model = load_model('model_19.h5', compile=False)
cap = genarate_desc(model, tokenizer, np.array([train_features['105342180_4d4a40b47f']]), maxlen)
print(cap)