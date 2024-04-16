import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk

nltk.download('punkt')
nltk.download('wordnet')

from pathlib import Path
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()


def prepare_data(intents, punctuations):
    """
    Prepare data for training a chatbot model.

    Args:
    - intents (dict): A dictionary containing intent patterns and tags.
    - punctuations (list): A list of punctuations to be excluded from words.

    Returns:
    - words (list): A list of lemmatized words in the dataset.
    - classes (list): A list of unique intent tags.
    - documents (list): A list of tuples containing tokenized words and corresponding intent tags.
    """
    words = []
    classes = []
    documents = []

    # Iterate over each intent in the intents dictionary
    for intent in intents['intents']:
        # Iterate over each pattern in the current intent
        for pattern in intent['patterns']:
            # Tokenize the pattern into a list of words
            wordList = nltk.word_tokenize(pattern)
            # Extend the words list with the tokenized words
            words.extend(wordList)
            # Append a tuple of wordList and intent tag to the documents list
            documents.append((wordList, intent['tag']))
            # Add the intent tag to the classes list if not already present
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize words and remove punctuations
    words = [lemmatizer.lemmatize(word) for word in words if word not in punctuations]
    # Sort and remove duplicates from the words list
    words = sorted(set(words))

    # Sort and remove duplicates from the classes list
    classes = sorted(set(classes))

    return words, classes, documents


def train_prep(words, classes, documents):
    """
    Prepare training data for a chatbot model.

    Args:
    - words (list): A list of lemmatized words in the dataset.
    - classes (list): A list of unique intent tags.
    - documents (list): A list of tuples containing tokenized words and corresponding intent tags.

    Returns:
    - trainX (numpy.ndarray): Input training data (bag of words).
    - trainY (numpy.ndarray): Output training data (one-hot encoded intent tags).
    """
    training = []
    outputEmpty = [0] * len(classes)

    # Iterate over each document in the documents list
    for document in documents:
        bag = []
        # Get the tokenized words from the document
        wordPatterns = document[0]
        # Lemmatize and convert words to lowercase
        wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
        # Create a bag of words representation
        for word in words:
            bag.append(1) if word in wordPatterns else bag.append(0)

        # Create one-hot encoded output row
        outputRow = list(outputEmpty)
        outputRow[classes.index(document[1])] = 1
        # Append bag of words and output row to training list
        training.append(bag + outputRow)

    # Shuffle the training data
    random.shuffle(training)
    # Convert training list to numpy array
    training = np.array(training)

    # Split the training data into input (bag of words) and output (intent tags) arrays
    trainX = training[:, :len(words)]
    trainY = training[:, len(words):]

    return trainX, trainY


def train_network(X_train, y_train, model_file_path):
    """
    Train a neural network model for intent classification.

    Args:
    - X_train (numpy.ndarray): Input training data (bag of words).
    - y_train (numpy.ndarray): Output training data (one-hot encoded intent tags).
    - model_file_path (str): File path to save the trained model.

    Returns:
    None
    """
    # Define the neural network model architecture
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(y_train[0]), activation='softmax'))

    # Configure the optimizer
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

    # Save the trained model
    model.save(model_file_path)
    print('Done')


def evaluate_model(X, y, model):
    # Evaluate accuracy
    loss, accuracy = model.evaluate(X, y)
    print('Accuracy:', accuracy)
    print('Loss:', loss)

    # Predict classes
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y, axis=1)

    # Generate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print('Confusion Matrix:')
    print(cm)

    # Generate classification report
    print('Classification Report:')
    print(classification_report(y_true_classes, y_pred_classes))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = [str(i) for i in range(y.shape[1])]  # assuming class labels are integers
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()


def main():
    top_dir = Path(__name__).absolute().parent
    intents = json.loads(open(f'{top_dir}/demo_data/demo_intents.json').read())

    with open(f'{top_dir}/data/punctuations.txt', 'r') as file:
        punctuations = file.read().splitlines()

    words, classes, documents = prepare_data(intents, punctuations)

    pickle.dump(words, open(f'{top_dir}/demo_data/words.pkl', 'wb'))
    pickle.dump(classes, open(f'{top_dir}/demo_data/classes.pkl', 'wb'))

    X_train, y_train = train_prep(words, classes, documents)
    # train_network(X_train, y_train, 'demo_fourChan_model.keras')

    model = load_model(f'{top_dir}/demo_fourChan_model.keras')
    evaluate_model(X_train, y_train, model)


if __name__ == "__main__":
    main()
