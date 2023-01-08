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
