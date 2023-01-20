import json
import os
import random
import fire
import cohere
from weblm.controllers.basic.pick_action import pick_action
from weblm.controllers.basic.pick_command import generate_command
from weblm.controllers.basic.prioritize import generate_prioritization
from weblm.controllers.basic.utils import CLICKABLE, MAX_NUM_ELEMENTS, TYPEABLE, DialogueState
from concurrent.futures import ThreadPoolExecutor

co = cohere.Client(os.environ.get("COHERE_KEY"), check_api_key=False)


def main():
    test_slice = split_history()
    os.chdir("/tmp")
    test_prioritization(test_slice)
    test_action_selection(test_slice)
    test_element_selection(test_slice)


def test_prioritization(history):
    top_1, top_5, top_10, top_20, top_40 = 0, 0, 0, 0, 0

    def _test(h):
        page_elements = list(filter(lambda x: any(x.startswith(y) for y in CLICKABLE + TYPEABLE), h["elements"]))
        prioritized_elements = generate_prioritization(co, h["objective"], page_elements, h["url"],
                                                       h["previous_commands"])
        correct_element = " ".join(h["command"].split(" ")[1:3])

        top_1_score = int(any(correct_element in e for e in prioritized_elements[:1]))
        top_5_score = int(any(correct_element in e for e in prioritized_elements[:5]))
        top_10_score = int(any(correct_element in e for e in prioritized_elements[:10]))
        top_20_score = int(any(correct_element in e for e in prioritized_elements[:20]))
        top_40_score = int(any(correct_element in e for e in prioritized_elements[:40]))

        return (top_1_score, top_5_score, top_10_score, top_20_score, top_40_score)

    with ThreadPoolExecutor(len(history)) as pp:
        results = pp.map(_test, history)

    for a, b, c, d, e in results:
        top_1 += a
        top_5 += b
        top_10 += c
        top_20 += d
        top_40 += e

    print("Prioritization Test")
    print(f"Top-1 Accuracy: {top_1 / len(history)}")
    print(f"Top-5 Accuracy: {top_5 / len(history)}")
    print(f"Top-10 Accuracy: {top_10 / len(history)}")
    print(f"Top-20 Accuracy: {top_20 / len(history)}")
    print(f"Top-40 Accuracy: {top_40 / len(history)}")


def test_action_selection(history):
    top_1 = 0

    def _test(h):
        page_elements = list(filter(lambda x: any(x.startswith(y) for y in CLICKABLE + TYPEABLE), h["elements"]))

        prioritized_elements = generate_prioritization(co, h["objective"], page_elements, h["url"],
                                                       h["previous_commands"])[:MAX_NUM_ELEMENTS]

        _, action, _ = pick_action(co, DialogueState.Action, None, h["objective"], h["url"], prioritized_elements,
                                   h["previous_commands"], None)

        correct_action = h["command"].split(" ")[0]

        return int(action.strip() == correct_action)

    with ThreadPoolExecutor(len(history)) as pp:
        results = pp.map(_test, history)

    for a in results:
        top_1 += a

    print("Action Selection Test")
    print(f"Top-1 Accuracy: {top_1 / len(history)}")


def test_element_selection(history):
    top_1, top_5, top_10, top_20, top_40 = 0, 0, 0, 0, 0

    def _test(h):
        page_elements = list(filter(lambda x: any(x.startswith(y) for y in CLICKABLE + TYPEABLE), h["elements"]))
        prioritized_elements = generate_prioritization(co, h["objective"], page_elements, h["url"],
                                                       h["previous_commands"])[:MAX_NUM_ELEMENTS]
        action = " " + h["command"].split(" ")[0]

        _, _, chosen_elements, _ = generate_command(co, DialogueState.Command, action, None, None, h["objective"],
                                                    h["url"], prioritized_elements, h["previous_commands"], None)

        chosen_elements = [e["id"] for e in chosen_elements]
        correct_element = " ".join(h["command"].split(" ")[1:3])

        top_1_score = int(any(correct_element in e for e in chosen_elements[:1]))
        top_5_score = int(any(correct_element in e for e in chosen_elements[:5]))
        top_10_score = int(any(correct_element in e for e in chosen_elements[:10]))
        top_20_score = int(any(correct_element in e for e in chosen_elements[:20]))
        top_40_score = int(any(correct_element in e for e in chosen_elements[:40]))

        return (top_1_score, top_5_score, top_10_score, top_20_score, top_40_score)

    with ThreadPoolExecutor(len(history)) as pp:
        results = pp.map(_test, history)

    for a, b, c, d, e in results:
        top_1 += a
        top_5 += b
        top_10 += c
        top_20 += d
        top_40 += e

    print("Command Element Selection Test")
    print(f"Top-1 Accuracy: {top_1 / len(history)}")
    print(f"Top-5 Accuracy: {top_5 / len(history)}")
    print(f"Top-10 Accuracy: {top_10 / len(history)}")
    print(f"Top-20 Accuracy: {top_20 / len(history)}")
    print(f"Top-40 Accuracy: {top_40 / len(history)}")


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