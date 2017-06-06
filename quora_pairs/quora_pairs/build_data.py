#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import re
import codecs
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation


import tensorflow as tf
keras=tf.contrib.keras.preprocessing

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

BASE_DIR="./data/"
TRAIN_DATA_FILE=BASE_DIR+"train.csv"
TEST_DATA_FILE=BASE_DIR+"test.csv"
VALIDATION_SPLIT=0.2
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH=15

##########################
#process texts in datasets
##########################
# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=True, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    #text = text.encode('utf-8')
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

#########################
# training data {seq1,seq2,label}
#########################
texts_1=[]
texts_2=[]
labels = []

with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in tqdm(reader):
        texts_1.append(text_to_wordlist(values[3]))
        texts_2.append(text_to_wordlist(values[4]))
        labels.append(int(values[5]))
print('Found %s texts in train.csv' % len(texts_1))
###########################
# test data {seq1,seq2}
###########################
test_texts_1 = []
test_texts_2 = []
test_ids = []
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in tqdm(reader):
        test_texts_1.append(text_to_wordlist(values[1]))
        test_texts_2.append(text_to_wordlist(values[2]))
        test_ids.append(values[0])
print('Found %s texts in test.csv' % len(test_texts_1))

## tf version >= 1.1?
tokenizer = keras.text.Tokenizer(num_words=MAX_NB_WORDS)

#https://github.com/fchollet/keras/issues/1072
# from unidecode import unidecode
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

#Transforms each text in texts in a sequence of integers.
sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))
nb_words = min(MAX_NB_WORDS, len(word_index))+1
print("the number of words:",nb_words) #120501

print("padding dataset and save to disk...")
#TODO: add sequence length.
# np.clip(a,min,max)
def leng(var):
    m = len(var)
    if m>=MAX_SEQUENCE_LENGTH:
        return MAX_SEQUENCE_LENGTH
    elif m<=1:
        return 1
    else:
        return m
data_1_len = np.array(map(leng,sequences_1))
data_1 = keras.sequence.pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH,
                                    padding='post',truncating='post')
data_2_len = np.array(map(leng,sequences_2))
data_2 = keras.sequence.pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH,
                                    padding='post',truncating='post')
labels = np.array(labels)
print('Shape of data tensor (seq1):', data_1.shape) #(404290,20)
print('Shape of data_len tensor (seq1_len):', data_1_len.shape)#(404290,)
print('Shape of data tensor (seq2):', data_2.shape)#(404290,20)
print('Shape of data_len tensor (seq2_len):', data_2_len.shape)#(404290,)
print('Shape of label tensor:', labels.shape)#(404290,)
##########Save to disk##############
if not tf.gfile.IsDirectory(BASE_DIR+"train/"):
    tf.gfile.MakeDirs(BASE_DIR+"train/")
np.save(BASE_DIR+'train/seq1',data_1)
np.save(BASE_DIR+'train/seq2',data_2)
np.save(BASE_DIR+'train/seq1_len',data_1_len)
np.save(BASE_DIR+'train/seq2_len',data_2_len)
np.save(BASE_DIR+'train/labels',labels)
print("training data Saved to: ",BASE_DIR+"train")

test_data_1_len = np.array(map(leng,test_sequences_1))
test_data_1 = keras.sequence.pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH,
                                        padding='post',truncating='post')
test_data_2_len = np.array(map(leng,test_sequences_2))
test_data_2 = keras.sequence.pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH,
                                        padding='post',truncating='post')
test_ids = np.array(test_ids)

if not tf.gfile.IsDirectory(BASE_DIR+"test/"):
    tf.gfile.MakeDirs(BASE_DIR+"test/")
np.save(BASE_DIR+'test/seq1',test_data_1)
np.save(BASE_DIR+'test/seq2',test_data_2)
np.save(BASE_DIR+'test/seq1_len',test_data_1_len)
np.save(BASE_DIR+'test/seq2_len',test_data_2_len)
np.save(BASE_DIR+'test/test_ids',test_ids)
print("training data Saved to: ",BASE_DIR+"test")

print("Save  Done!!")

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

print("save word_index!!")
np.save('data/word_index',np.array(word_index))
# word_index = np.load('data/word_index.npy')

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
#####################
np.save('data/embedding_map',embedding_matrix)
print("Embedding save done!!")

