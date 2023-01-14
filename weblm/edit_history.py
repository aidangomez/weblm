"""Search the examples.json file and delete any bad entries."""

import json
import os
import re

import cohere
import numpy as np

co = cohere.Client(os.environ.get("COHERE_KEY"))


def search_history(query, history):
    embeds = [h["embedding"] for h in history]
    examples = [h["example"] for h in history]
    embeds = np.array(embeds)
    embedded_state = np.array(co.embed(texts=[query], truncate="RIGHT").embeddings[0])
    scores = np.einsum("i,ji->j", embedded_state,
                       embeds) / (np.linalg.norm(embedded_state) * np.linalg.norm(embeds, axis=1))
    ind = np.argsort(scores)
    return np.array(examples)[ind], ind


if __name__ == "__main__":
    with open("examples.json", "r") as fd:
        history = json.load(fd)

    indices_for_deletion = []

    for h in history:
        if "objective" not in h:
            # print(h.keys())
            # print(h["example"])

            example = h["example"]
            match = re.search(r"Objective: ([\w\.\,\?\!\$ ]+)\n", example)
            if match:
                objective = match.group(1)
                h["objective"] = objective

    with open("examples_tmp.json", "w") as fd:
        json.dump(history, fd)
    os.replace("examples_tmp.json", "examples.json")
