import random
import json
import pickle
import numpy as np
import os

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

# Global Variables/Objects
lemmatizer = WordNetLemmatizer()

words = pickle.load(open(f'chan_model{os.path.sep}demo_data{os.path.sep}words.pkl', 'rb'))
classes = pickle.load(open(f'chan_model{os.path.sep}demo_data{os.path.sep}classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    """
    Preprocess a sentence for intent classification.

    Args:
    - sentence (str): Input sentence to be preprocessed.

    Returns:
    - sentence_words (list): A list of lemmatized and lowercase tokenized words.
    """
    # Tokenize the input sentence into a list of words
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize and convert words to lowercase
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    """
    Convert a sentence into a bag of words representation.

    Args:
    - sentence (str): Input sentence to be converted.

    Returns:
    - numpy.ndarray: A numpy array representing the bag of words.
    """
    # Preprocess the sentence to obtain individual words
    sentence_words = clean_up_sentence(sentence)
    # Initialize a bag of words with zeros, with length equal to the total number of words in the dataset
    bag = [0] * len(words)
    # Iterate over each word in the preprocessed sentence
    for w in sentence_words:
        # Iterate over each word in the dataset
        for i, word in enumerate(words):
            # If the word in the dataset matches the current word in the sentence, set the corresponding index in the bag to 1
            if word == w:
                bag[i] = 1
    # Convert the bag of words list into a numpy array and return it
    return np.array(bag)


class ChanModel:

    def __init__(self):
        # self.model = load_model(model_nane)
        self.model = load_model(f'chan_model{os.path.sep}demo_fourChan_model.keras')
        self.intents = json.loads(open(f'chan_model{os.path.sep}demo_data{os.path.sep}demo_intents.json').read())

    def predict_class(self, sentence):
        """
        Predict the intent of a given sentence using a trained model.

        Args:
        - sentence (str): Input sentence for intent classification.

        Returns:
        - list: A list of dictionaries containing predicted intents and their probabilities.
        """
        # Convert the sentence into a bag of words representation
        bow = bag_of_words(sentence)
        # Use the trained model to predict the intent probabilities for the input bag of words
        res = self.model.predict(np.array([bow]))[0]
        # Set a threshold to filter out intents with probabilities below a certain value
        ERROR_THRESHOLD = 0.25
        # Filter out intents with probabilities above the threshold
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        # Sort the results by probability in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        # Initialize an empty list to store the predicted intents and their probabilities
        return_list = []
        # Iterate over the filtered results
        for r in results:
            # Append a dictionary containing the predicted intent and its probability to the return list
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        # Return the list of predicted intents and their probabilities
        return return_list

    def get_response(self, intents_list):
        """
        Get a response based on the predicted intent.

        Args:
        - intents_list (list): List of dictionaries containing predicted intents and their probabilities.
        - intents_json (dict): JSON object containing predefined intents and responses.

        Returns:
        - str: A response corresponding to the predicted intent.
        """
        # Extract the predicted intent from the intents list
        tag = intents_list[0]['intent']
        # Retrieve the list of predefined intents and responses from the intents JSON
        list_of_intents = self.intents['intents']
        # Iterate over each predefined intent
        for i in list_of_intents:
            # If the tag of the current predefined intent matches the predicted intent
            if i['tag'] == tag:
                # Randomly choose a response from the list of responses associated with the matched intent
                result = random.choice(i['responses'])
                break
        # Return the selected response
        return result


if __name__ == "__main__":
    print("Bot ONLINE")
    chan_model = ChanModel()
    while True:
        message = input("")
        ints = chan_model.predict_class(message)
        res = chan_model.get_response(ints)
        print(res)
