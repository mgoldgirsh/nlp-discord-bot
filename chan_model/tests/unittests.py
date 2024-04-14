import unittest
import json
import os
from test_functions import prepare_data, clean_up_sentence, bag_of_words
import numpy as np


class TestPrepareData(unittest.TestCase):
    def test_prepare_data(self):
        # First set of intents
        intents_1 = {
            "intents": [
                {"tag": "greeting", "patterns": ["Hi", "Hello!"]},
                {"tag": "goodbye", "patterns": ["Bye", "Nice talking with you", "cya"]}
            ]
        }
        punctuations_1 = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '{', '|', '}', '~']

        expected_words_1 = ['Bye', 'Hello', 'Hi', 'Nice', 'cya', 'talking', 'with', 'you']
        expected_classes_1 = ['goodbye', 'greeting']
        expected_docs_1 = [(['Hi'], 'greeting'), (['Hello', '!'], 'greeting'), (['Bye'], 'goodbye'), (['Nice', 'talking', 'with', 'you'], 'goodbye'), (['cya'], 'goodbye')]

        words_1, classes_1, documents_1 = prepare_data(intents_1, punctuations_1)

        self.assertListEqual(words_1, expected_words_1)
        self.assertListEqual(classes_1, expected_classes_1)
        self.assertListEqual(documents_1, expected_docs_1)

        # Second set of intents
        intents_2 = {
            "intents": [
                {"tag": "options", "patterns": ["How you could help me?", "What you can do?", "What help you provide?", "How you can be helpful?", "Anything else?"]},
                {"tag": "political", "patterns": ["What do you think about Trump?", "Should I vote for Trump?"]}
            ]
        }
        punctuations_2 = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '{', '|', '}', '~']

        expected_words_2 = ['Anything', 'How', 'I', 'Should', 'Trump', 'What', 'about', 'be', 'can', 'could', 'do', 'else', 'for', 'help', 'helpful', 'me', 'provide', 'think', 'vote', 'you']
        expected_classes_2 = ['options', 'political']
        expected_docs_2 = [(['How', 'you', 'could', 'help', 'me', '?'], 'options'), (['What', 'you', 'can', 'do', '?'], 'options'), (['What', 'help', 'you', 'provide', '?'], 'options'), (['How', 'you', 'can', 'be', 'helpful', '?'], 'options'), (['Anything', 'else', '?'], 'options'), (['What', 'do', 'you', 'think', 'about', 'Trump', '?'], 'political'), (['Should', 'I', 'vote', 'for', 'Trump', '?'], 'political')]

        words_2, classes_2, documents_2 = prepare_data(intents_2, punctuations_2)

        self.assertListEqual(words_2, expected_words_2)
        self.assertListEqual(classes_2, expected_classes_2)
        self.assertListEqual(documents_2, expected_docs_2)


class TestCleanUpSentence(unittest.TestCase):
    def test_clean_up_sentence(self):
        # With some spelling error
        input_sentence_1 = "Give me someapples!"
        expected_output_1 = ['give', 'me', 'someapples', '!']
        output_1 = clean_up_sentence(input_sentence_1)
        self.assertEqual(output_1, expected_output_1)

        # With longer sentences
        input_sentence_2 = "I am going to the park."
        expected_output_2 = ['i', 'am', 'going', 'to', 'the', 'park', '.']
        output_2 = clean_up_sentence(input_sentence_2)
        self.assertEqual(output_2, expected_output_2)

        # Another case
        input_sentence_3 = "Do you like ice cream?"
        expected_output_3 = ['do', 'you', 'like', 'ice', 'cream', '?']
        output_3 = clean_up_sentence(input_sentence_3)
        self.assertEqual(output_3, expected_output_3)


class TestBagOfWords(unittest.TestCase):

    def test_bag_of_words(self):
        # With all caps
        input_sentence1 = "HOW ARE YOU"
        expected_output1 = np.array([0, 1, 0, 0, 1])
        self.assertTrue(np.array_equal(bag_of_words(input_sentence1), expected_output1))
        
        # With punctuation
        input_sentence2 = "How are you?"
        expected_output2 = np.array([0, 1, 0, 0, 1])
        self.assertTrue(np.array_equal(bag_of_words(input_sentence2), expected_output2))

        # With lower and upper cases
        input_sentence3 = "HoW arE yoU"
        expected_output3 = np.array([0, 1, 0, 0, 1])
        self.assertTrue(np.array_equal(bag_of_words(input_sentence3), expected_output3))

        # With words that do not appear in existing word list
        input_sentence4 = "hello there!"
        expected_output4 = np.array([0, 0, 0, 0, 0])
        self.assertTrue(np.array_equal(bag_of_words(input_sentence4), expected_output4))


if __name__ == '__main__':
    unittest.main()
