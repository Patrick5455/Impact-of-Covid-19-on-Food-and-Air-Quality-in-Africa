# utility
import re
import os
import logging
import string
import pickle
import itertools
import numpy as np
import pandas as pd
import preprocessor as p
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# word2vec
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nltk.download('stopwords')

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_sIZE = 0.8

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

SEQUENCE_lENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024

KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"

decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

stopWords = stopwords.words("english")
stemmer = SnowballStemmer("english")

def decode_sentiment(label):
    return decode_map[int(label)]

# remove links, users, and special characters
def preprocess(text, stem=False):
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stopWords:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)

    return " ".join(tokens)

def cleanTweets(twitterText):
    print("Cleaning Tweets :) \n\n")
    tweet = p.clean(twitterText)

    # Happy Emoticons
    emoticonsHappy = set([
        ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3'])

    # Sad Emoticons
    emoticonsSad = set([
        ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
        ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
        ':c', ':{', '>:\\', ';('])

    # Emoji patterns
    emojiPattern = re.compile("["u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                 u"\U00002702-\U000027B0"
                                 u"\U000024C2-\U0001F251" "]+", flags=re.UNICODE)

    # Combine the sad and happy emoticons
    emoticons = emoticonsHappy.union(emoticonsSad)

    # tokenize the words in the tweet and get the stop words
    stopWords = set(stopwords.words('english'))
    wordTokens = nltk.word_tokenize(tweet)

    # after tweepy preprocessing the colon symbol left remain after
    # removing mentions
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)

    # replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)

    # remove emojis from tweet
    tweet = emojiPattern.sub(r'', tweet)

    # looping through conditions
    filteredTweet = []
    for w in wordTokens:
        # check tokens against stop words , emoticons and punctuations
        if w not in stopWords and w not in emoticons and w not in string.punctuation:
            filteredTweet.append(w)

    return ' '.join(filteredTweet)

def word2vecModel(docs):
    w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, window=W2V_WINDOW, min_count=W2V_MIN_COUNT,
                                                workers=8)
    w2v_model.build_vocab(docs)

    words = w2v_model.wv.vocab.keys()
    vocabSize = len(words)
    print("Vocab Size is: ", vocabSize)

    w2v_model.train(docs, total_examples=len(docs), epochs=W2V_EPOCH)

    return w2v_model


def tokenizerTings(dfTrain, dfTest):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dfTrain.text)

    vocab_size = len(tokenizer.word_index) + 1
    print("Total words", vocab_size)

    xTrain = pad_sequences(tokenizer.texts_to_sequences(dfTrain.text), maxlen=SEQUENCE_lENGTH)
    xTest = pad_sequences(tokenizer.texts_to_sequences(dfTest.text), maxlen=SEQUENCE_lENGTH)

    return tokenizer, vocab_size, xTrain, xTest

def encodeLabels(dfTrain, dfTest):
    encoder = LabelEncoder()
    encoder.fit(dfTrain.target.tolist())

    yTrain = encoder.transform(dfTrain.target.tolist())
    yTest = encoder.transform(dfTest.target.tolist())

    yTrain = yTrain.reshape(-1, 1)
    yTest = yTest.reshape(-1, 1)

    return yTrain, yTest

def embeddingManenos(vocabSize):
    embeddingMatrix = np.zeros((vocabSize, W2V_SIZE))
    for word, i in tokeniza.word_index.items():
        if word in w2vModel.wv:
            embeddingMatrix[i] = w2vModel.wv[word]

    print("embedding Matrix shape:", embeddingMatrix.shape)

    embeddingLayer = Embedding(vocabSize, W2V_SIZE, weights=[embeddingMatrix], input_length=SEQUENCE_lENGTH,
                               trainable=False)

    return embeddingLayer

def nnModel(embeddingLayer):
    model = Sequential()
    model.add(embeddingLayer)
    model.add(Dropout(0.5))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

    model.summary()

    return model


if __name__ == "__main__":

    df = pd.read_csv('data/training.1600000.processed.noemoticon.csv',
                     encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
    df.text = df.text.apply(lambda x: cleanTweets(x))

    dfTrain, dfTest = train_test_split(df, test_size=1 - TRAIN_sIZE, random_state=42)

    docs = [_text.split() for _text in dfTrain.text]

    w2vModel = word2vecModel(docs)

    tokeniza, vocabSize, xTrain, xTest = tokenizerTings(dfTrain, dfTest)
    yTrain, yTest = encodeLabels(dfTrain, dfTest)

    embeddingLayer = embeddingManenos(vocabSize)

    callbacks = [ReduceLROnPlateau(monitor="val_loss", patience=5, cooldown=0),
                 EarlyStopping(monitor="val_acc", min_delta=1e-4, patience=5)]

    model = nnModel(embeddingLayer)
    history = model.fit(xTrain, yTrain, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1,
                        verbose=1, callbacks=callbacks)

    score = model.evaluate(xTest, yTest, batch_size=BATCH_SIZE)
    print()
    print("ACCURACY:", score[1])
    print("LOSS:", score[0])

    model.save(KERAS_MODEL)
    w2vModel.save(WORD2VEC_MODEL)
    pickle.dump(tokeniza, open(TOKENIZER_MODEL, "wb"), protocol=0)
