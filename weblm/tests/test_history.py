import json
import os
import random
import fire
import requests
import cohere
import numpy as np
from weblm.controllers.basic.pick_action import pick_action
from weblm.controllers.basic.pick_command import CommandGeneration
from weblm.controllers.basic.prioritize import generate_prioritization
from weblm.controllers.basic.utils import CLICKABLE, MAX_NUM_ELEMENTS, TYPEABLE, DialogueState
from concurrent.futures import ThreadPoolExecutor

co = cohere.Client(os.environ.get("COHERE_KEY"), check_api_key=False)


def main():
    test_slice = split_history()
    os.chdir("/tmp")
    test_prioritization_action_command(test_slice)


def test_prioritization_action_command(history):

    def _test(h):
        while True:
            try:
                page_elements = list(filter(lambda x: any(x.startswith(y) for y in CLICKABLE + TYPEABLE),
                                            h["elements"]))

                prioritized_elements = generate_prioritization(co, h["objective"], page_elements, h["url"],
                                                               h["previous_commands"])[:MAX_NUM_ELEMENTS]

                correct_element = " ".join(h["command"].split(" ")[1:3])
                prioritization_top_1_score = int(any(correct_element in e for e in prioritized_elements[:1]))
                prioritization_top_5_score = int(any(correct_element in e for e in prioritized_elements[:5]))
                prioritization_top_10_score = int(any(correct_element in e for e in prioritized_elements[:10]))
                prioritization_top_20_score = int(any(correct_element in e for e in prioritized_elements[:20]))
                prioritization_top_40_score = int(any(correct_element in e for e in prioritized_elements[:40]))
                scores = [
                    prioritization_top_1_score, prioritization_top_5_score, prioritization_top_10_score,
                    prioritization_top_20_score, prioritization_top_40_score
                ]

                _, action, _ = pick_action(co, DialogueState.Action, None, h["objective"], h["url"],
                                           prioritized_elements, h["previous_commands"], None)

                correct_action = h["command"].split(" ")[0]

                action_acc = int(action.strip() == correct_action)
                scores.append(action_acc)

                if correct_action == "click":
                    prioritized_elements = list(
                        filter(lambda x: any(x.startswith(y) for y in CLICKABLE), prioritized_elements))
                elif correct_action == "type":
                    prioritized_elements = list(
                        filter(lambda x: any(x.startswith(y) for y in TYPEABLE), prioritized_elements))
                else:
                    assert 0, correct_action

                _, _, chosen_elements, _ = CommandGeneration().generate_command(co, DialogueState.Command,
                                                                                " " + correct_action, None, None,
                                                                                h["objective"], h["url"],
                                                                                prioritized_elements,
                                                                                h["previous_commands"], None)

                chosen_elements = [e["id"] for e in chosen_elements]
                correct_element = " ".join(h["command"].split(" ")[1:3])

                command_top_1_score = int(any(correct_element in e for e in chosen_elements[:1]))
                command_top_5_score = int(any(correct_element in e for e in chosen_elements[:5]))
                command_top_10_score = int(any(correct_element in e for e in chosen_elements[:10]))
                command_top_20_score = int(any(correct_element in e for e in chosen_elements[:20]))
                command_top_40_score = int(any(correct_element in e for e in chosen_elements[:40]))
                scores += [
                    command_top_1_score, command_top_5_score, command_top_10_score, command_top_20_score,
                    command_top_40_score
                ]

                return np.array(scores)
            except (cohere.CohereError, requests.exceptions.RetryError):
                print("cohere fucked up, retrying...")

    with ThreadPoolExecutor(len(history)) as pp:
        results = pp.map(_test, history)

    scores = np.array([0] * 11)
    for a in results:
        scores += a

    print("Prioritization Test")
    print(f"Top-1 Accuracy: {scores[0] / len(history)}")
    print(f"Top-5 Accuracy: {scores[1] / len(history)}")
    print(f"Top-10 Accuracy: {scores[2] / len(history)}")
    print(f"Top-20 Accuracy: {scores[3] / len(history)}")
    print(f"Top-40 Accuracy: {scores[4] / len(history)}")

    print("Action Selection Test")
    print(f"Top-1 Accuracy: {scores[5] / len(history)}")

    print("Command Element Selection Test")
    print(f"Top-1 Accuracy: {scores[6] / len(history)}")
    print(f"Top-5 Accuracy: {scores[7] / len(history)}")
    print(f"Top-10 Accuracy: {scores[8] / len(history)}")
    print(f"Top-20 Accuracy: {scores[9] / len(history)}")
    print(f"Top-40 Accuracy: {scores[10] / len(history)}")


def split_history():
    with open("examples.json", "r") as fd:
        history = json.load(fd)

    objective_keyed_history = {}

    required_keys = ["objective", "url", "previous_commands", "command"]
    history = list(filter(lambda h: all(k in h for k in required_keys), history))

    # sort by objective
    for h in history:
        if h["objective"] in objective_keyed_history:
            objective_keyed_history[h["objective"]].append(h)
        else:
            objective_keyed_history[h["objective"]] = [h]

    # split 80/20
    test_percent = 0.2
    test_slice_length = int(test_percent * len(objective_keyed_history))

    print(f"History length: {len(history)}")
    print(f"Total distinct objectives: {len(objective_keyed_history)}")

    objectives = list(objective_keyed_history.keys())
    random.seed(2311)
    random.shuffle(objectives)
    test_slice = objectives[:test_slice_length]
    test_slice = [h for k in test_slice for h in objective_keyed_history[k]]
    memory_slice = objectives[test_slice_length:]
    memory_slice = [h for k in memory_slice for h in objective_keyed_history[k]]

    print(f"Test slice: {len(test_slice)}")
    print(f"Memory slice: {len(memory_slice)}")

    # write to /tmp
    with open("/tmp/examples.json", "w+") as fd:
        json.dump(memory_slice, fd)

    return test_slice


if __name__ == "__main__":
    fire.Fire(main)