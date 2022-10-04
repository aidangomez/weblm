import asyncio
import chunk
import os
from typing import Dict, Tuple

import cohere
import discord
from discord import Embed
from discord.ext import commands
from playwright.async_api import async_playwright

from .controller import Command, Controller, Prompt
from .crawler import AsyncCrawler

co = cohere.Client(os.environ.get("COHERE_KEY"))


def chunk_message_for_sending(msg):
    chunks = []
    tmp_chunk = ""
    for line in msg.split("\n"):
        if len(tmp_chunk + line) > 2000:
            chunks.append(tmp_chunk)
            tmp_chunk = line
        else:
            tmp_chunk += "\n" + line

    if tmp_chunk != "":
        chunks.append(tmp_chunk)

    return chunks


class MyClient(discord.Client):

    def __init__(self, playwright, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sessions: Dict[int, Tuple[Crawler, Controller]] = {}
        self.playwright = playwright

    async def on_ready(self):
        """Initializes bot"""
        print(f"We have logged in as {self.user}")

        for guild in self.guilds:
            print(f"{self.user} is connected to the following guild:\n"
                  f"{guild.name}(id: {guild.id})")

    async def find_session(self, id, message):
        print(message.clean_content)
        objective = message.clean_content

        if id not in self.sessions:
            print("did not find session")
            crawler, controller = (AsyncCrawler(self.playwright), Controller(co, objective))
            await crawler._init_browser()
            print("browser inited")
            self.sessions[id] = (crawler, controller)
            await crawler.go_to_page("google.com")
            print("crawler on page")

        crawler, controller = self.sessions[id]

        return (crawler, controller)

    async def respond_to_message(self, message):
        print(message.clean_content)
        objective = message.clean_content
        crawler, controller = await self.find_session(message.id, message)

        if objective == "cancel":
            del self.sessions[message.id]
            return

        while True:
            content = await crawler.crawl()
            print("AIDAN", content)

            async with message.channel.typing():
                if not controller.is_running():
                    print("Controller not yet running")
                    response = controller.dialogue_step(crawler.page.url, content)
                else:
                    response = controller.dialogue_step(crawler.page.url, content, response=objective)

                print(response)

                if isinstance(response, Command):
                    print("running command", response)
                    await crawler.run_cmd(str(response))
                elif isinstance(response, Prompt):
                    thread = await message.create_thread(name=objective)
                    for chunk in chunk_message_for_sending(str(response)):
                        msg = await thread.send(chunk)
                        await msg.edit(suppress=True)
                    return

    async def respond_in_thread(self, message):
        if message.channel.starter_message.id not in self.sessions:
            print("Session not running, killing")
            await message.channel.send("This session is dead please begin a new one in #web-lm.")
            return

        print(message.clean_content)
        objective = message.clean_content
        crawler, controller = await self.find_session(message.channel.starter_message.id, message)

        if objective == "cancel":
            del self.sessions[message.channel.starter_message.id]
            return

        while True:
            content = await crawler.crawl()
            print("AIDAN", content)

            async with message.channel.typing():
                if not controller.is_running():
                    print("Controller not yet running")
                    response = controller.dialogue_step(crawler.page.url, content)
                else:
                    response = controller.dialogue_step(crawler.page.url, content, response=objective)

                print(response)

                if isinstance(response, Command):
                    print("running command", response)
                    await crawler.run_cmd(str(response))
                elif isinstance(response, Prompt):
                    for chunk in chunk_message_for_sending(str(response)):
                        msg = await message.channel.send(chunk)
                        await msg.edit(suppress=True)
                    return

    async def respond_to_dm(self, message):
        print(message.clean_content)
        objective = message.clean_content
        crawler, controller = await self.find_session(message.author.id, message)

        if objective == "cancel":
            del self.sessions[message.author.id]
            return

        while True:
            content = await crawler.crawl()
            print("AIDAN", content)

            async with message.channel.typing():
                if not controller.is_running():
                    print("Controller not yet running")
                    response = controller.dialogue_step(crawler.page.url, content)
                else:
                    response = controller.dialogue_step(crawler.page.url, content, response=objective)

                print(response)

                if isinstance(response, Command):
                    print("running command", response)
                    await crawler.run_cmd(str(response))
                elif isinstance(response, Prompt):
                    for chunk in chunk_message_for_sending(str(response)):
                        msg = await message.channel.send(chunk)
                        await msg.edit(suppress=True)
                    return

    async def on_message(self, message):
        print(message)
        if isinstance(
                message.channel,
                discord.TextChannel) and message.channel.id == 1026557845308723212 and message.author != self.user:
            await self.respond_to_message(message)
        elif isinstance(message.channel, discord.DMChannel) and message.author != self.user:
            await self.respond_to_dm(message)
        elif isinstance(
                message.channel,
                discord.Thread) and message.channel.parent.id == 1026557845308723212 and message.author != self.user:
            await self.respond_in_thread(message)


async def main():
    intents = discord.Intents.all()
    async with async_playwright() as playwright:
        async with MyClient(playwright, intents=intents) as client:
            try:
                await client.start(os.environ.get("DISCORD_KEY"))
            except Exception as e:
                print(f"Exception caught: {e}")


if __name__ == "__main__":
    asyncio.run(main())
