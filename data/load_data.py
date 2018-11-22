import pandas as pd
import numpy as np
import os
import csv
from keras import preprocessing
from keras.preprocessing.text import Tokenizer
DATA_DIR = os.path.dirname(os.path.realpath(__file__))
STS_DIR = os.path.join(DATA_DIR, 'stsbenchmark')
GLOVE_DIR  = os.path.join(DATA_DIR, 'glove6B')

def get_snli():
    colnames = ['sentence1', 'sentence2', 'isSim'] 
    df = pd.read_csv(DATA_DIR + '/train_snli.txt',sep="	",header=None, names=colnames)
    df['sentence1']= df['sentence1'].astype(str)
    df['sentence2']= df['sentence2'].astype(str)
    return df

def split_df(df, proportion=0.8):
    msk = np.random.rand(len(df)) < proportion
    train = df[msk]
    test = df[~msk]
    return train,test

def load_sts():
    colnames = ['1','genre', 'filename', 'year', 'score', 'sentence1', 'sentence2', 'other'] 
    keep = ['score', 'sentence1', 'sentence2']
    remove = list(set(colnames) - set(keep))
    def load_file(name):
        df = pd.read_csv(os.path.join(STS_DIR, 'sts-{}.csv'.format(name)),sep="	",header=None,
                        names=colnames, quoting=csv.QUOTE_NONE, encoding='utf-8')
        return df.drop(remove, axis=1)
    return tuple(map(lambda x: load_file(x), ['train', 'dev', 'test']))


def get_tokenizer(df_arr, max_words=10000):
    sentences = []
    for df in df_arr:
        sentences = sentences + list(df["sentence1"].values) + list(df["sentence2"].values)
    sentences = np.unique(sentences)
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(sentences)
    return tokenizer

def get_embedding_matrix(word_index, max_words, embedding_dim=100):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def tokenize_df(df, tokenizer, maxlen=20):
    pad = preprocessing.sequence.pad_sequences
    x1 =  pad(tokenizer.texts_to_sequences(df['sentence1']), maxlen=maxlen)
    x2 =  pad(tokenizer.texts_to_sequences(df['sentence2']), maxlen=maxlen)
    return x1,x2