import discord

from conversational_model.discord_model import DiscordModel


class NLPBot(discord.Client):

    def __init__(self, intents, discord_bot_model: DiscordModel) -> None:
        """
        Initializes the discord bot with given intents and the model to use
        for natural language processing outputs.
        :param intents: the intents to use with the discord bot
        :param discord_bot_model: the model to use to generate the responses for the bot
        :return None, this step initializes the variables for the discord bot to operate correctly
        """
        super().__init__(intents=intents)
        self.discord_bot_model = discord_bot_model
        self.previous_user = ""

    async def on_ready(self) -> None:
        """
        Runs when the discord bot is ready to receive commands.
        :return: None, prints that the bot is ready/connected.
        """
        print(f'{self.user} has connected to Discord!')

    async def on_message(self, message) -> None:
        """
        On a message received in the discord chat perform this action.
        :param message: the message received in the discord chat.
        :return: None, sends a corresponding message to the same channel the message was sent in.
        """

        if message.author == self.user:
            return

        # the previous author of the message is not the same as the one that said the sentence before
        # then clear the history of that conversation with the discord bot
        if message.author != self.previous_user:
            self.discord_bot_model.clear_history()
            self.previous_user = message.author

        # the discord bot receives the message content and generates the response
        async with message.channel.typing():
            out_response = self.discord_bot_model.generate(message.content)

        await message.channel.send(out_response)
