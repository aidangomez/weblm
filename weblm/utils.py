HELP_MSG = """Welcome to WebLM!

The goal of this project is build a system that takes an objective from the user, and operates a browser to carry it out.

For example:
- book me a table for 2 at bar isabel next wednesday at 7pm
- i need a flight from SF to London on Oct 15th nonstop
- buy me more whitening toothpaste from amazon and send it to my apartment

WebLM learns to carry out tasks *by demonstration*. That means that you'll need to guide it and correct it when it goes astray. Over time, the more people who use it, the more tasks it's used for, WebLM will become better and better and rely less and less on user input.

To control the system:
- You can see what the model sees at each step by looking at the list of elements the model can interact with
- show: You can also see a picture of the browser window by typing `show`
- goto: You can go to a specific webpage by typing `goto www.yourwebpage.com`
- success: When the model has succeeded at the task you set out (or gotten close enough), you can teach the model by typing `success` and it will save it's actions to use in future interations
- cancel: If the model is failing or you made a catastrophic mistake you can type `cancel` to kill the session
- help: Type `help` to show this message

Everytime you use WebLM it will improve. If you want to contribute to the project and help us build join the discord (https://discord.com/invite/co-mmunity) or send an email to weblm@cohere.com"""


class Prompt:

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt

    def __str__(self) -> str:
        return self.prompt


class Command:

    def __init__(self, cmd: str) -> None:
        self.cmd = cmd

    def __str__(self) -> str:
        return self.cmd
