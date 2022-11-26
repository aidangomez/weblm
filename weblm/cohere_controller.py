import csv
import heapq
import itertools
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, DefaultDict, Dict, List, Tuple, Union

import cohere
import numpy as np
from requests.exceptions import ConnectionError
import weblm.config as config

from weblm.templates import *


class CohereController:

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
            objective (str): the objective to  accomplish
        """
        self.co = co
        self.objective = objective
        self.previous_commands: List[str] = []
        self.moments: List[Tuple[str, str, str]] = []
        self.user_responses: DefaultDict[str, int] = defaultdict(int)


    def search(self, query: str, items: List[str], topk: int) -> List[str]:
        embedded_items = np.array(self.co.embed(texts=items, truncate="RIGHT").embeddings)
        embedded_query = np.array(self.co.embed(texts=[query], truncate="RIGHT").embeddings[0])
        scores = np.einsum("i,ji->j", embedded_query,
                        embedded_items) / (np.linalg.norm(embedded_query) * np.linalg.norm(embedded_items, axis=1))
        ind = np.argsort(scores)[-topk:]
        return np.flip(np.array(items)[ind], axis=0)

    def truncate_left(self, prompt, *rest_of_prompt, limit=2048):
        i = 0
        chop_size = 5
        print(f"WARNING: truncating sequence of length {len(self.co.tokenize(prompt + ''.join(rest_of_prompt)))} to length {limit}")
        while len(self.co.tokenize(prompt + "".join(rest_of_prompt))) > limit:
            prompt = prompt[i * chop_size:]
            i += 1
        return prompt


    def _get_cmd_prediction(self, action, prompt: str, chosen_element: str) -> str:
            if "type" in action:
                text = None
                while text is None:
                    try:
                        num_tokens = 20
                        if len(self.co.tokenize(prompt)) > 2048 - num_tokens:
                            print(f"WARNING: truncating sequence of length {len(self.co.tokenize(prompt))}")
                            prompt = self.truncate_left(
                                                prompt,
                                                action,
                                                chosen_element,
                                                limit=2048 - num_tokens)

                        print(len(self.co.tokenize(prompt + action + chosen_element)))
                        text = max(self.co.generate(prompt=prompt + action + chosen_element,
                                                    model=config.MODEL,
                                                    temperature=0.5,
                                                    num_generations=5,
                                                    max_tokens=num_tokens,
                                                    stop_sequences=["\n"],
                                                    return_likelihoods="GENERATION").generations,
                                key=lambda x: x.likelihood).text
                        print(text)
                    except cohere.error.CohereError as e:
                        print(f"Cohere fucked up: {e}")
                        continue
            else:
                text = ""

            return (action + chosen_element + text).strip()


    def _fn(self, x):
        if len(x) == 3:
            option, prompt, self = x
            return_likelihoods = "ALL"
        elif len(x) == 4:
            option, prompt, self, return_likelihoods = x

        while True:
            try:
                if len(self.co.tokenize(prompt)) > 2048:
                    prompt = self.truncate_left(prompt)
                return (self.co.generate(prompt=prompt, max_tokens=0, model=config.MODEL,
                                        return_likelihoods=return_likelihoods).generations[0].likelihood, option)
            except cohere.error.CohereError as e:
                print(f"Cohere fucked up: {e}")
                continue
            except ConnectionError as e:
                print(f"Connection error: {e}")
                continue

    def choose(self,
                template: str,
                options: List[Dict[str, str]],
                return_likelihoods: str = "ALL",
                topk: int = 1) -> List[Tuple[int, Dict[str, str]]]:
            """Choose the most likely continuation of `prompt` from a set of `options`.

            Args:
                template (str): a string template with keys that match the dictionaries in `options`
                options (List[Dict[str, str]]): the options to be chosen from

            Returns:
                str: the most likely option from `options`
            """
            num_options = len(options)
            with ThreadPoolExecutor(num_options) as pp:
                _lh = pp.map(
                    self._fn,
                    zip(options, [template.format(**option) for option in options], [self] * num_options,
                        [return_likelihoods] * num_options))
            return sorted(_lh, key=lambda x: x[0], reverse=True)[:topk]