import csv
import os
import re
from collections import defaultdict


MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
          "december")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
SEASONS = ("winter", "fall", "spring", "summer")

NEGATIONS = {}
VERBALIZATION = defaultdict(dict)
ROLESETS = {}
CATEGORIES = {}


def read_resources():
    prev_dir = os.getcwd()
    if read_resources.done:
        return
    try:
        os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources"))
        with open("negations.txt", encoding="utf-8") as f:
            NEGATIONS.update(csv.reader(f, delimiter=" "))
        with open("rolesets.txt", encoding="utf-8") as f:
            ROLESETS.update((l[0], tuple(l[1:])) for l in csv.reader(f))
        lines = []
        with open("wordnet.txt", encoding="utf-8") as f:
            lines += [re.findall(r'(\S):(\S+)', l) for l in f if l]
        with open("morph-verbalization-v1.01.txt", encoding="utf-8") as f:
            lines += [re.findall(r'::DERIV\S*-(\S)\S+ "(\S+)"', l) for l in f if l and l[0] != "#"]
        for pairs in lines:
            for prefix, word in pairs:
                VERBALIZATION[word].update(pairs)
        with open("verbalization-list-v1.06.txt", encoding="utf-8") as f:
            lines = (re.findall(r"(\S+) TO *(\S+ :\S+)? (\S+-\d+) *(\S+)?", l)[0] for l in f if l and l[0] not in "#D")
            for word, category, verb, suffix in lines:
                VERBALIZATION[word]["V"] = verb
                if category or suffix:
                    CATEGORIES[word] = category.replace(" ", "") + suffix
        with open("have-org-role-91-roles-v1.06.txt", encoding="utf-8") as f:
            # noinspection PyTypeChecker
            CATEGORIES.update(l.split()[::-1] for l in f if l and l[0] not in "#")
        with open("have-rel-role-91-roles-v1.06.txt", encoding="utf-8") as f:
            CATEGORIES.update(re.findall(r"(\S+) (\S+(?: [^:#]\S)*)", l)[0][::-1] for l in f if l and l[0] not in "#")
    finally:
        os.chdir(prev_dir)
    read_resources.done = True


read_resources.done = False
