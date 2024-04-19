from conversational_model.dataset import DialogDataset
from conversational_model.tokens import Token


def test_tokenize_data():
    dataset = DialogDataset("daily_dialog", "openai-community/gpt2")
    sample_data = {
        "act": [0, 1],
        "dialog": ["Hello, what is your name", "My name is Bob"]
    }
    assert dataset.tokenize_data(sample_data) == [f"{Token.USER_TOKEN} Hello, what is your name "
                                                  f"{Token.COMPUTER_TOKEN} My name is Bob"]


if __name__ == "__main__":
    test_tokenize_data()
