import csv
import json
import os
import re
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple, Union

import cohere
from weblm.utils import Command, Prompt, HELP_MSG


class Controller:
    """A Cohere-powered controller that takes in a browser state and produces and action.

    The basic outline of this Controller's strategy is:
    1. receive page content from browser
    2. prioritise elements on page based on how relevant they are to the objective
    3. look up similar states from the past
    4. choose between clicking and typing
    5. choose what element to click or what element to type in
    """

    def __init__(self, co: cohere.Client, objective: str):
        """
        Args:
            co (cohere.Client): a Cohere Client
            objective (str): the objective to accomplish
        """
        self.co = co
        self.objective = objective
        self.previous_commands: List[str] = []
        self.moments: List[Tuple[str, str, str, List[str]]] = []
        self.user_responses: DefaultDict[str, int] = defaultdict(int)
        self.reset_state()

    def is_running(self):
        return self._step != DialogueState.Unset

    def reset_state(self):
        self._step = DialogueState.Unset
        self._action = None
        self._cmd = None
        self._chosen_elements: List[Dict[str, str]] = []
        self._prioritized_elements = None
        self._pruned_prioritized_elements = None
        self._prioritized_elements_hash = None
        self._page_elements = None
        self._error = None

    def success(self):
        for url, elements, command, previous_commands in self.moments:
            self._save_example(objective=self.objective,
                               url=url,
                               elements=elements,
                               command=command,
                               previous_commands=previous_commands)

    def _save_example(self, objective: str, url: str, elements: List[str], command: str, previous_commands: List[str]):
        state = construct_state(objective, url, elements[:MAX_NUM_ELEMENTS], previous_commands)
        example = ("Example:\n"
                   f"{state}\n"
                   f"Next Command: {command}\n"
                   "----")
        print(f"Example being saved:\n{example}")
        with open("examples.json", "r") as fd:
            history = json.load(fd)
            examples = [h["example"] for h in history]

        if example in examples:
            print("example already exists")
            return

        history.append({
            "example": example,
            "embedding": self.co.embed(texts=[example]).embeddings[0],
            "url": url,
            "elements": elements,
            "command": command,
            "previous_commands": previous_commands,
            "objective": objective,
        })

        with open("examples_tmp.json", "w") as fd:
            json.dump(history, fd)
        os.replace("examples_tmp.json", "examples.json")

    def _construct_responses(self):
        keys_to_save = ["y", "n", "s", "command", "success", "cancel"]
        responses_to_save = defaultdict(int)
        for key, value in self.user_responses.items():
            if key in keys_to_save:
                responses_to_save[key] = value
            elif key not in keys_to_save and key:
                responses_to_save["command"] += 1

        self.user_responses = responses_to_save
        print(f"Responses being saved:\n{dict(responses_to_save)}")

    def save_responses(self):
        keys_to_save = ["y", "n", "s", "command", "success", "cancel"]
        # Check if data file already exists
        responses_filepath = "responses.csv"
        if os.path.isfile(responses_filepath):
            print("File exists")
            with open(responses_filepath, "a+") as fd:
                wr = csv.writer(fd, quoting=csv.QUOTE_ALL)
                wr.writerow([self.user_responses[key] for key in keys_to_save])
        else:
            print("No data available")
            with open(responses_filepath, "w+") as fd:
                wr = csv.writer(fd, quoting=csv.QUOTE_ALL)
                wr.writerow(keys_to_save)
                wr.writerow([self.user_responses[key] for key in keys_to_save])

    def step(self, url: str, page_elements: List[str], response: str = None) -> Union[Prompt, Command]:
        pass
