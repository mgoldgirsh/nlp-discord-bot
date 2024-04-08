import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

from tokens import Token


class DialogDataset(Dataset):
    """
    Represents a dataset that keeps track of the dialog  associated with it
    for training.
    """

    def __init__(self, dataset_name: str, tokenizer_name: str):
        dataset = load_dataset(dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.train_data = dataset['train']
        self.train_tokenized_data = self.tokenize_data(self.train_data)

        self.encoded_dialogs = self.encode_dialogs()

    def tokenize_data(self, loaded_data: dict) -> list:
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

    def encode_dialogs(self):
        return self.tokenizer(self.train_tokenized_data, padding=True, truncation=True, return_tensors='pt')

    def __len__(self):
        return len(self.train_tokenized_data)

    def __getitem__(self, idx: int):
        return {
            'input_ids': self.encoded_dialogs['input_ids'][idx],
            'attention_mask': self.encoded_dialogs['attention_mask'][idx],
            'labels': self.encoded_dialogs['input_ids'][idx].type(torch.LongTensor)
        }
