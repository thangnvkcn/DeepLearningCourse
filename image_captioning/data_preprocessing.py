import pandas as pd
from keras.preprocessing.image import load_img
data = pd.read_csv('data/results.csv',  delimiter='\t')
data = data.values

print(data[0])
#
# filename = data[:, 0]
# y_train = data[:, 1]
#
# filename = [filename[i].split('.')[0] for i in range(len(filename))]
#
# descriptions = dict()
#
# for i in range(len(filename)):
#     if filename[i] not in descriptions:
#         descriptions[filename[i]] = list()
#     descriptions[filename[i]].append(y_train[i])
#
# #Cleaning data
# # print(descriptions)
#
# import string
#
# punctuations = string.punctuation
# table = str.maketrans('', '', string.punctuation)
# vocab = set()
# vocab_count = dict()
# for key, desc_file in descriptions.items():
#     for i in range(len(desc_file)):
#         desc_file[i] = desc_file[i].lower()
#         words = desc_file[i].split()
#         stripped = [w.translate(table) for w in words]
#         desc_file[i] = str.strip(' '.join(stripped))
#
#
#
# for key, desc_file in descriptions.items():
#     for i in range(len(desc_file)):
#         vocab.update(desc_file[i].split())
#         for k in desc_file[i].split():
#             if k not in vocab_count:
#                 vocab_count[k] = 0
#             vocab_count[k] += 1
# for key in vocab_count:
#     if vocab_count[key] > 10:
#         vocab.remove(key)
#
