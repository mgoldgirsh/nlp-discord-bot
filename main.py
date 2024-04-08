import discord
from pathlib import Path

from bot.utils import read_token
from bot.bot import NLPBot
from conversational_model.discord_model import DiscordModel

if __name__ == "__main__":
    # setup the token for discord bot api
    TOKEN = read_token()

    # setup the intents for the discord bot
    intents = discord.Intents.default()
    intents.message_content = True

    # setup the model to use in the discord bot
    model = DiscordModel(Path(__name__).absolute().parent)

    # setup the discord bot with the model
    client = NLPBot(intents, model)
    client.run(TOKEN)
