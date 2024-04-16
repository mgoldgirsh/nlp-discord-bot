import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

from tokens import Token


class DialogDataset(Dataset):
    """
    Represents a dataset that keeps track of the dialog associated with it
    for training.
    """

    def __init__(self, dataset_name: str, tokenizer_name: str):
        """
        Initialize the dialog dataset with the dataset name ("dialog_dataset") and the tokenizer name
        "openai/gpt2" in order to create a formatted dialog dataset for generative learning.

        :param dataset_name: the dialog dataset to use to obtain data for training generative ai
        :param tokenizer_name: the tokenizer to use when encoding sentences for the generative model
        """
        dataset = load_dataset(dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.train_tokenized_data = self.tokenize_data(dataset['train'])
        self.validation_tokenized_data = self.tokenize_data(dataset['validation'])

        self.encoded_dialogs = self.encode_dialogs(self.train_tokenized_data)
        self.test_encoded_dialogs = self.encode_dialogs(self.validation_tokenized_data)

    def tokenize_data(self, loaded_data: dict) -> list:
        """
        Takes a dictionary of loaded data from the hugging face dataset and prefixes tokens onto the dataset.
        Ex.
            [["Hello, the weather is nice today."], ["Yes, it is"]]
            "<|USER|> Hello, the weather is nice today. <|COMPUTER|> Yes, it is”

        :param loaded_data: the dictionary of utterances to prefix with user and computer tokens
        :return: the tokenized sentence with the tokens before the utterances
        """
        tokenized_data = []
        for datum in loaded_data:
            # first check whether there are exactly 2 users in the conversation
            if len(set(datum['act'])) != 2:
                continue

            # now loop through dialog and actors to get tokenized conversations
            tokenized_str = f"{Token.USER_TOKEN}"
            previous_actor = datum['act'][0]
            for dialog, actor in zip(datum['dialog'], datum['act']):
                if actor == previous_actor:
                    tokenized_str += f" {dialog}"
                else:
                    tokenized_str += f" {Token.COMPUTER_TOKEN} {dialog}"

                previous_actor = actor

            # now add the eos token to the end of the tokenized sentence
            tokenized_str += f" {self.tokenizer.eos_token}"
            tokenized_data.append(tokenized_str.strip())
        return tokenized_data

    def encode_dialogs(self, tokenized_data):
        """
        Encodes the prefixed dialogs with byte-pair encoding.

        :param tokenized_data: the prefix tokenized data to encode with byte pair
        :return: the byte pair encoded vectors of the tokenization operation
        """
        return self.tokenizer(tokenized_data, padding=True, truncation=True, return_tensors='pt')

    def __len__(self) -> int:
        """
        Returns the length of the encoded data.

        :return: the int length of the encoded data
        """
        return len(self.train_tokenized_data)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns the encoded train dialog at specified idx.

        :param idx: the int index to get the item at
        :return: the dict of encoded dialogs at that idx
        """
        return {
            'input_ids': self.encoded_dialogs['input_ids'][idx],
            'attention_mask': self.encoded_dialogs['attention_mask'][idx],
            'labels': self.encoded_dialogs['input_ids'][idx].type(torch.LongTensor)
        }
