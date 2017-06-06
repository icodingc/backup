#coding=utf-8
from tqdm import tqdm
import numpy as np

print("initialize embedding matrix")
#################initialize embedding matrix..
embeddings_index = {}
f = open('data/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

word_index = np.load('data/word_index.npy')

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
#####################
np.save('data/embedding_map',embedding_matrix)
print("Embedding save done!!")