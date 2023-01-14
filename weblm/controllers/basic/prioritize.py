"""The goal of Prioritization is to sort and filter the elements on a webpage so that the *most relevant* elements to the objective at hand are shown to the model."""

import random
from typing import List
import numpy as np
import cohere
import json

from weblm.controllers.basic.utils import construct_state, choose

prioritization_template = """$examples
---
Here are the most relevant elements on the webpage (links, buttons, selects and inputs) to achieve the objective below:
Objective: $objective
URL: $url
Relevant elements:
{element}"""

priorit_tmp = ("Objective: {objective}"
               "\nURL: {url}"
               "\nRelevant elements:"
               "\n{elements}")


def gather_prioritisation_examples(co: cohere.Client, state: str, topk: int = 6, num_elements: int = 3) -> List[str]:
    """Simple semantic search over a file of past interactions to find the most similar ones."""
    with open("examples.json", "r") as fd:
        history = json.load(fd)

    if len(history) == 0:
        return []

    embeds = [h["embedding"] for h in history]
    examples = [h["example"] for h in history]
    embeds = np.array(embeds)
    embedded_state = np.array(co.embed(texts=[state], truncate="RIGHT").embeddings[0])
    scores = np.einsum("i,ji->j", embedded_state,
                       embeds) / (np.linalg.norm(embedded_state) * np.linalg.norm(embeds, axis=1))
    ind = np.argsort(scores)[-topk:]

    ind = ind.tolist()
    prioritisation_examples = [None] * len(ind)
    for i, h in enumerate(history):
        if i in ind:
            if all(x in h for x in ["objective", "command", "url", "elements"]):
                # make sure the element relevant to the next command is included
                elements = list(filter(lambda x: x[:4] != "text", h["elements"]))
                command_element = " ".join(h["command"].split()[1:3])
                command_element = list(filter(lambda x: command_element in x, elements))
                assert len(command_element) == 1, f"length is {len(command_element)}"
                command_element = command_element[0]

                if not command_element in elements[:num_elements]:
                    rand_idx = random.randint(0, num_elements - 1)
                    elements = elements[:rand_idx] + [command_element] + elements[rand_idx:]

                elements = elements[:num_elements]

                objective = h["objective"]
                url = h["url"]
                elements = '\n'.join(elements)
                prioritisation_example = eval(f'f"""{priorit_tmp}"""')
                prioritisation_examples[ind.index(i)] = prioritisation_example

    return list(filter(lambda x: x is not None, prioritisation_examples))


def generate_prioritization(co: cohere.Client, objective: str, page_elements: List[str], url: str,
                            previous_commands: List[str]):
    state = construct_state(objective, url, page_elements, previous_commands)
    examples = gather_prioritisation_examples(co, state)

    prioritization = prioritization_template
    prioritization = prioritization.replace("$examples", "\n---\n".join(examples))
    prioritization = prioritization.replace("$objective", objective)
    prioritization = prioritization.replace("$url", url)

    print(prioritization)

    prioritized_elements = choose(co, prioritization, [{"element": x} for x in page_elements], topk=len(page_elements))
    prioritized_elements = [x[1]["element"] for x in prioritized_elements]

    return prioritized_elements
