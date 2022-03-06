import os
from tqdm import tqdm

tags = []
words = []
spacy_text = ""


def tag(w, is_entity=True):
    if len(w) == 1 and is_entity:
        return ["U"]
    elif len(w) == 1:
        return ["O"]
    if is_entity:
        return ["B"] + ["I"] * (len(w) - 2) + ["L"]
    return ["O"] * len(w)


def formatWord(word, tag):
    pad = len(word) // 2
    if len(word) % 2 == 1:
        return " " * pad + tag + " " * pad
    return " " * pad + tag + " " * (pad - 1)


def clr():
    os.system("cls" if os.name == "nt" else "clear")


def status():
    print("spacy thinks....")
    print(spacy_text)
    print()
    print()
    print("What do you think?")
    print()
    print()
    print(" ".join(statusText))
    print(" ".join([formatWord(x, y) for x, y in zip(statusText, tags)]))
    print()


def annotateText(text):
    t = text.split()
    choices = {"y", "n", "b", "restart", "", "break"}
    e_count = 0
    global tags
    tags = []
    i = 0
    while i < len(t):
        w = t[i]
        isEnt = None
        status()
        while isEnt not in choices:
            isEnt = input(f"Is '{w}' part of an entity?{choices}: ")

            if isEnt == "break":
                if e_count != 0:
                    tags.extend(tag(t[max(i - e_count, 0) : max(i, e_count)], True))
                    clr()
                    status()
                e_count = 0
                isEnt = None

        if isEnt == "b":
            i -= 1
            i -= e_count
            i = max(0, i)
            e_count = 0
            tags = tags[:i]
            clr()
            continue

        if isEnt == "restart":
            return annotateText(text)
        if isEnt == "y" and len(t) - 1 != i:
            e_count += 1
        else:
            if isEnt == "y":  # if the last entity is an entity
                e_count += 1
            if e_count != 0:
                tags.extend(tag(t[max(i - e_count, 0) : max(i, e_count)], True))
            if isEnt in ("n", ""):
                tags.extend(tag(t[i : i + 1], False))
            e_count = 0
        clr()
        i += 1

    assert len(tags) == len(t)
    return tags


if __name__ == "__main__":
    import pandas as pd
    import spacy

    nlp = spacy.load("en_core_web_lg")

    data = pd.read_csv("annotated.csv")
    if "tags" not in data.columns:
        data["tags"] = None
    bar = tqdm(data.iterrows(), total=len(data))
    for i, row in bar:
        clr()
        bar.refresh()
        print()
        if not pd.isnull(row["tags"]):
            continue
        doc = nlp(row["0"])
        spacy_text = "\n".join(
            [f"{ent.text} {ent.label_} {spacy.explain(ent.label_)}" for ent in doc.ents]
        )
        statusText = row["0"].split()
        statusTags = []

        tags = annotateText(row["0"])
        data.at[i, "tags"] = tags
        data.to_csv("annotated.csv")
