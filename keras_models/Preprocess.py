import numpy as np
import pandas as pd
import glob
import os
import sys
import pickle
import fasttext
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import KFold
import configuration as config


class Preprocessing:

    # handle attributes and attribute
    def preprocess(self, docs, labels=None, attribute=None, stats=True):
        # delete '_' because I want name to be concatenated
        t = Tokenizer(num_words=20000, filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
        t.fit_on_texts(docs)
        encoded_docs = t.texts_to_sequences(docs)
        if stats:
            print("BEFORE Pruning:")
            self.get_statistics(encoded_docs, labels, attribute)
        idx2word = {v: k for k, v in t.word_index.items()}

        # stopwords
        stopwrd = set(stopwords.words('english'))

        # handle abbreviation
        def abbreviation_handler(text):
            ln = text.lower()
            # case replacement
            ln = ln.replace(r"'t", " not")
            ln = ln.replace(r"'s", " is")
            ln = ln.replace(r"'ll", " will")
            ln = ln.replace(r"'ve", " have")
            ln = ln.replace(r"'re", " are")
            ln = ln.replace(r"'m", " am")

            # delete single '
            ln = ln.replace(r"'", " ")

            return ln

        # handle stopwords
        def stopwords_handler(text):
            words = text.split()
            new_words = [w for w in words if w not in stopwrd]
            return ' '.join(new_words)

        # get post-tokenized docs
        def sequence_to_text(listOfSequences):
            tokenized_list = []
            for text in listOfSequences:
                newText = ''
                for num in text:
                    newText += idx2word[num] + ' '
                newText = abbreviation_handler(newText)
                newText = stopwords_handler(newText)
                tokenized_list.append(newText)
            return tokenized_list

        newLists = sequence_to_text(encoded_docs)

        return newLists

    # handle single and multiple attributes
    def get_statistics(self, encoded_docs, Ys=None, attributes=None):

        # explore encoded docs: find sequence length distribution
        result = [len(x) for x in encoded_docs]
        print("Min=%d, Mean=%d, Max=%d" % (np.min(result), np.mean(result), np.max(result)))
        data_max_seq_length = np.max(result) + 1

        # explore Y for each attribute
        if type(attributes) == list:
            for idx, attribute in enumerate(attributes):
                Y = Ys[:, idx]
                self.get_single_statistics(Y, attribute)
        elif type(attributes) == str:
            self.get_single_statistics(Ys, attributes)

        return data_max_seq_length

    # get statistics of a single attribute

    def get_single_statistics(self, Y, attribute):

        # find majority distribution
        ct = Counter(Y)
        majorityDistribution = max([ct[k] * 100 / float(Y.shape[0]) for k in ct])
        print("Total majority is {0} for {1}.".format(majorityDistribution, attribute))

    def load_text(self, text):
        # preprocess data before feeding into tokenizer
        docs = self.preprocess([text], stats=False)

        # tokenize the data
        t = Tokenizer(num_words=config.MAX_NUM_WORDS)
        t.fit_on_texts(docs)
        encoded_docs = t.texts_to_sequences(docs)
        # print("Real Vocab Size: %d" % (len(t.word_index) + 1))
        self.word_index = t.word_index

        # perform Bag of Words
        self.X = pad_sequences(encoded_docs, maxlen=config.MAX_SEQ_LENGTH)  # use either a fixed max-length or the real max-length from data

    def load_data(self, attribute):

        # read the data
        df = pd.read_csv(config.FILENAME)

        print("The size of data is {0}".format(df.shape[0]))
        docs = df[config.column_to_read].astype(str).values.tolist()
        labels = df[attribute].values  # attribute is either string or a list of strings
        # preprocess data before feeding into tokenizer
        docs = self.preprocess(docs, labels, attribute)

        # tokenize the data
        t = Tokenizer(num_words=config.MAX_NUM_WORDS)
        t.fit_on_texts(docs)
        encoded_docs = t.texts_to_sequences(docs)
        print("Real Vocab Size: %d" % (len(t.word_index) + 1))
        print("Truncated Vocab Size: %d" % config.MAX_NUM_WORDS)
        self.word_index = t.word_index

        # perform Bag of Words
        print("AFTER Pruning:")
        data_max_seq_length = self.get_statistics(encoded_docs, labels, attribute)  # can be used to replace MAX_SEQ_LENGTH
        self.X = pad_sequences(encoded_docs, maxlen=config.MAX_SEQ_LENGTH)  # use either a fixed max-length or the real max-length from data

        # for use with categorical_crossentropy
        self.Y = labels
        if config.multilabel:
            self.onehot_Y = self.Y
        else:
            self.onehot_Y = keras.utils.to_categorical(labels, config.ClassNum)

        self.attribute = attribute if type(attribute) == str else 'all_attributes'  # self.attribute is always a string

    def load_fasttext(self, use_word_embedding):

        # if not use it,
        if not use_word_embedding:
            self.embedding_matrix = []
            self.EMBEDDING_DIM = 100
            return

        # if exists
        if os.path.exists("fastMatrix.pkl"):
            print("FastText Embedding EXISTS!")
            with open("fastMatrix.pkl", 'rb') as f:
                embedding_matrix = pickle.load(f)
            self.embedding_matrix = embedding_matrix
            self.EMBEDDING_DIM = len(embedding_matrix[0])
            return

        # read the fastText Embedding
        count_oov = 0
        fastModel = fasttext.load_model(config.PATH_EMBEDDING)

        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((config.MAX_NUM_WORDS, config.EMBEDDING_DIM))
        word2idx = {k: v for k, v in self.word_index.items() if v < config.MAX_NUM_WORDS}  # ??? explore word_index, how to eliminate stop words directly from it if it ranks words in frequency
        for word, i in word2idx.items():
            embedding_matrix[i] = fastModel[word]  # a vector is giving oov words anyway
            if word not in fastModel.words:
                count_oov += 1
        print("Finished Vectorization.\n {} / {} Not in FastText Bin.".format(count_oov, config.MAX_NUM_WORDS))

        # update the variables
        self.embedding_matrix = embedding_matrix
        self.EMBEDDING_DIM = config.EMBEDDING_DIM

        # dump the matrix
        with open("fastMatrix.pkl", "wb") as f:
            pickle.dump(embedding_matrix, f)

        return

    # generater, returns indices for train and test data
    def cross_validation(self, X, Y, seed):
        # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        kfold = KFold(n_splits=config.kfolds, shuffle=True, random_state=seed)
        kfolds = kfold.split(X, Y)
        return kfolds
