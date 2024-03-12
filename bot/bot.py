import discord

from utils import read_token

# setup the token for discord bot api
TOKEN = read_token()

# setup the discord bot client
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    """
    Runs when the discord bot is ready to receive commands.
    :return: None, prints that the bot is ready/connected.
    """
    print(f'{client.user} has connected to Discord!')


@client.event
async def on_message(message):
    """
    On a message received in the discord chat perform this action.
    :param message: the message received in the discord chat.
    :return: None, sends a corresponding message to the same channel the message was sent in.
    """
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')


if __name__ == "__main__":
    client.run(TOKEN)
