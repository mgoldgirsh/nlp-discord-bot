from transformers import AutoTokenizer, AutoModelForCausalLM

from conversational_model.tokens import Token


class DiscordModel:
    def __init__(self, model_dir):
        """
        Initializes the discord model with directory where the model is located.

        :param model_dir: the directory location of where the model is located.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.discord_model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/discord_gpt_model")

        self.dialog = ""

    def generate(self, user_input: str, with_history: bool = True):
        """
        Generates an output based on the user_input sentence. Can generate the output based on historical inputs
        and outputs or not.

        :param user_input: the user input to generate the output from
        :param with_history: whether to use the historical inputs into the model
        :return: the generated output in str format to display
        """
        tokenized_user_input = f"{Token.USER_TOKEN} {user_input} "
        self.dialog += tokenized_user_input
        if with_history:
            print(self.dialog)
        conversation_idx = len(self.dialog.split(f'{Token.USER_TOKEN}')) - 1
        encoded_user_input = self.tokenizer(self.dialog, padding=True, truncation=True, return_tensors='pt')
        out_response = self.discord_model.generate(encoded_user_input['input_ids'],
                                                   max_length=min(1024 - encoded_user_input['input_ids'].size(-1), 100),
                                                   do_sample=True,
                                                   top_k=50,
                                                   eos_token_id=self.tokenizer.eos_token_id,
                                                   pad_token_id=self.tokenizer.pad_token_id)
        decoded_response = self.tokenizer.decode(out_response[0], skip_special_tokens=False)
        decoded_response = decoded_response.split(f"{Token.USER_TOKEN}")[conversation_idx]
        computer_idx = decoded_response.find(f"{Token.COMPUTER_TOKEN}")
        end_of_text_idx = decoded_response.find(f"{self.tokenizer.eos_token}")

        if computer_idx != -1:
            decoded_response = decoded_response.split(f"{Token.COMPUTER_TOKEN}")[1]
        if end_of_text_idx == -1:
            decoded_response = decoded_response.strip(self.tokenizer.eos_token)

        if with_history:
            self.dialog += f"{Token.COMPUTER_TOKEN} {decoded_response} "
            # make the dialog history 150 char long
            while len(self.dialog) > 150:
                start_idx = self.dialog.find(f"{Token.USER_TOKEN}")
                self.dialog = self.dialog[start_idx + len(Token.USER_TOKEN):]
        else:
            # reset the history of the dialog
            self.dialog = ""
        return decoded_response

    def clear_history(self):
        """
        Clears the historical dialog responses.

        :return: None, has the effect of clearing historical dialog responses.
        """
        self.dialog = ""
