from pathlib import Path
import os

current_directory = Path(__file__).absolute().parent


def read_token() -> str:
    """
    Reads the token from the "token.key" file and outputs the str
    representing the api token for the discord bot.

    :return: a str representing the api token for the discord bot.
    """
    with open(os.path.join(str(current_directory.parent), "token.key")) as f:
        token = f.read().strip()
    return token
