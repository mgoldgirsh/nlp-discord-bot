import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('../data/intents.json').read())

words = pickle.load(open('../data/words.pkl', 'rb'))
classes = pickle.load(open('../data/classes.pkl', 'rb'))
model = load_model('../fourChan_model.keras')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    words = ["Hi", "how", "I", "help", "you"]
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def prepare_data(intents, punctuations):
    words = []
    classes = []
    documents = []

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            wordList = nltk.word_tokenize(pattern)
            words.extend(wordList)
            documents.append((wordList, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(word) for word in words if word not in punctuations]
    words = sorted(set(words))

    classes = sorted(set(classes))

    return words, classes, documents