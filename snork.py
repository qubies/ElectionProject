## Imports and constants
import yaml
import re
import csv
import spacy
import random
import os
import argparse
import pandas as pd
import numpy as np
from random import choice
from enum import Enum, unique
from tqdm import tqdm
from collections import Counter
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from print_funcs import *
import configparser
from os import path


def remove_special_characters(text, keep_digits=True):
    if keep_digits:
        return kd_norm.sub(" ", repl.sub(" ", text))
    return rd_norm.sub(" ", repl.sub(" ", text))


def lemmatize(text):
    sent = []
    doc = nlp(text)
    for word in doc:
        if (
            word.is_punct == False
            and word.is_alpha == True
            and (not word.is_stop or word.text == "not")
            and word.ent_type_ != "PERSON"
        ):
            sent.append(word.lemma_)
    return " ".join(sent)


# the standardize fn from the original book
def standardize(text):
    return lemmatize(remove_special_characters(str(text).lower()))


# build a regex for a single keyword
def word_select_pattern(word, require_full_match=False):
    # partial allows partial matches starting from the wordbreaks
    # ("err" matches the ERRor etc)
    return re.compile(
        r"\b" + word.replace("_", " ") + (r"\b" * require_full_match), re.IGNORECASE
    )


# build a joint or pattern from a group of keys
def build_or_pattern(word_list, require_full_match=False):
    if require_full_match:
        return f"\\b({'|'.join(map(lambda x: x.replace('_', ' '), word_list))})\\b"
    return f"\\b({'|'.join(map(lambda x: x.replace('_', ' '), word_list))})"


# run on the df, sets a column to true if standardized text contains
def set_col_contains(col, word_list):
    global annotated_df
    pattern = re.compile(build_or_pattern(word_list))
    for index, row in annotated_df.iterrows():
        if pattern.search(row.standardized_text):
            if row.gold == None:
                row.gold = []
            row.gold.append(col.value)


def normalize(s):
    try:
        return (
            s.upper().strip().replace("\xa0", " ").replace(", ", "_").replace(" ", "_")
        )
    except:
        return "NULL"

    # LABELLING FUNCTIONS
    # Single word lf functions


def fn_from_topic_and_keyword(
    topic, keyword, require_full_match=False, standardized=False
):

    r = word_select_pattern(keyword, require_full_match)
    if standardized:
        text_field = "standardized_text"
    else:
        text_field = "raw_text"

    @labeling_function()
    def foo(text):
        try:
            if r.search(text[text_field].lower()):
                return topic.value
            return -1
        except:
            return -1

    foo.name = (
        f"{topic}_{keyword}_Full-{require_full_match}_Standardized-{standardized}"
    )
    return foo


# Multi-word LFS Definition
def fn_from_topic_and_wordlist(topic, words, standardized=False):

    if standardized:
        text_field = "standardized_text"
    else:
        text_field = "raw_text"

    @labeling_function()
    def foo(text):
        try:
            if all(x in text[text_field].lower() for x in words):
                return topic.value
            return -1
        except:
            return -1

    foo.name = f"{topic}_{'-and-'.join(words)}_Full-False_Standardized-{standardized}"
    return foo


# Multi-word LFS Definition
def fn_from_regex(topic, regexExpression, standardized=False):
    if standardized:
        text_field = "standardized_text"
    else:
        text_field = "raw_text"
    reg = re.compile(regexExpression, [re.IGNORECASE, re.MULTILINE])

    @labeling_function()
    def foo(text):
        try:
            m = reg.search(text)
            if len(m.groups) < 1:
                return -1
            else:
                return m.group
            if all(x in text[text_field].lower() for x in words):
                return topic.value
            return -1
        except:
            return -1

    foo.name = f"{topic}_{'-and-'.join(words)}_Full-False_Standardized-{standardized}"
    return foo


@print_banner_completion_wrapper("Topics", banner_token=" ")
def print_topics():
    for topic in TOPICS:
        print(topic.value, topic.name)
    assert TOPICS["ABSTAIN"].value == -1


def undecidable_length(length, standardized=False):
    if standardized:
        text_field = "standardized_text"
    else:
        text_field = "raw_text"

    @labeling_function()
    def foo(text):
        try:
            if len(text[text_field]) < length:
                return TOPICS.UNDECIDABLE.value
            return -1
        except:
            return (
                TOPICS.UNDECIDABLE.value
            )  # If we cant read it, its probably undecitable

    foo.name = (
        f"{TOPICS.UNDECIDABLE}_LENGTH-BASED_length-{length}_Standardized-{standardized}"
    )
    return foo


def standardize_df(df, text_field):
    if "standardized_text" not in df.columns:
        df["standardized_text"] = df[text_field].progress_apply(
            lambda x: standardize(x)
        )
        df["raw_text"] = df[text_field]


def get_probabilities(df, applier, model):
    print("Applying LFS")
    preds = [[], []]
    for x in range(0, len(df), 1024):
        print(f"Labelling up to {x}")
        a, b = model.predict(
            applier.apply(df=df[max(x - 1024, 0) : x]), return_probs=True
        )
        preds[0].extend(a)
        preds[1].extend(b)
    a, b = model.predict(applier.apply(df=df[x:]), return_probs=True)
    preds[0].extend(a)
    preds[1].extend(b)
    print(len(preds), len(df))
    assert len(preds[0]) == len(df)
    return (np.array(preds[0]), np.array(preds[1]))


def label_df(df, applier, model, text_field, standardize=True):
    if standardize:
        standardize_df(df, text_field)
    model_pred = get_probabilities(df, applier, model)

    labs = []
    labels, probabilities = model_pred
    min_prob = 1 / probabilities.shape[1] + 0.0001
    ties = 0
    for x in range(len(labels)):
        if labels[x] != -1:
            labs.append(TOPICS(labels[x]).name)
        elif np.max(probabilities[x]) > min_prob:
            breaker = choice(np.where(probabilities[x] == np.max(probabilities[x]))[0])
            labs.append(TOPICS(breaker).name)
            ties += 1
        else:
            labs.append(TOPICS.UNDECIDABLE.name)

    df["snorkel_prediction"] = labs
    for topic in TOPICS:
        df[f"probability_topic_{topic.name}"] = [x[topic.value] for x in probabilities]
    print(f"Number of ties encountered: {ties}")


# global stuff....
repl = re.compile(r"\|’|\u2019|\u2018")
kd_norm = re.compile(r"[^a-zA-z0-9\s]")
rd_norm = re.compile(r"[^a-zA-z\s]")
pd.set_option("max_seq_item", None)
pd.set_option("max_columns", None)
pd.set_option("max_rows", None)

# sanity check
# tests.....
assert word_select_pattern("hat", True).search("hatrack") == None
assert word_select_pattern("hat", False).search("hatrack") != None
TOPICS = Enum("TOPICS", "UNDECIDABLE")
mock_df = {"standardized_text": "hatrack", "raw_text": "hatrack"}
test_fn = fn_from_topic_and_keyword(TOPICS.UNDECIDABLE, "hat", True, True)
assert test_fn(mock_df) == -1
test_fn = fn_from_topic_and_keyword(TOPICS.UNDECIDABLE, "hat", True, False)
assert test_fn(mock_df) == -1
mock_df = {"standardized_text": "hat", "raw_text": "hat"}
assert test_fn(mock_df) == TOPICS.UNDECIDABLE.value
test_fn = fn_from_topic_and_keyword(TOPICS.UNDECIDABLE, "hat", True, True)
assert test_fn(mock_df) == TOPICS.UNDECIDABLE.value

mock_df = {"standardized_text": "money introduced", "raw_text": "money in"}
test_fn = fn_from_topic_and_wordlist(TOPICS.UNDECIDABLE, ["money", "introduced"], True)
assert test_fn(mock_df) == TOPICS.UNDECIDABLE.value
test_fn = fn_from_topic_and_wordlist(TOPICS.UNDECIDABLE, ["money", "introduced"], False)
assert test_fn(mock_df) == -1

if __name__ == "__main__":

    # setup....
    tqdm.pandas()  # progressssss
    nlp = spacy.load("en_core_web_sm")  # load spacy module

    config = configparser.ConfigParser()
    config.read("config.ini")

    SNORKEL_to_label_paths = [
        x.strip() for x in config.get("INPUTS", "SNORKEL_DATA_TO_LABEL").split(",")
    ]
    for p in SNORKEL_to_label_paths:
        assert path.exists(p)
    assert path.exists(config.get("INPUTS", "SNORKEL_TRAINING_DATA").strip())
    assert path.exists(config.get("INPUTS", "ANNOTATED_DATA").strip())
    assert path.exists(config.get("SNORKEL_LABELLING", "SINGLE_KEYS_FILE").strip())
    assert path.exists(config.get("SNORKEL_LABELLING", "MULTI_KEYS_FILE").strip())

    # lazy sets
    text_column = config.get("INPUT_DETAILS", "TEXT_COLUMN").strip()
    label_column = config.get("INPUT_DETAILS", "LABEL_COLUMN").strip()

    with PrintContext("Loading Data"):
        # set the topics
        with open(config.get("SNORKEL_LABELLING", "TOPICS_FILE")) as infile:
            TOPICS = yaml.load(infile, Loader=yaml.Loader)

        TOPICS = unique(
            Enum(
                "TOPICS",
                [topic.upper().replace(" ", "_") for topic in TOPICS],
                start=-1,  # ABSTAIN has to be -1
            )
        )

        print_topics()

        with open(config.get("SNORKEL_LABELLING", "SINGLE_KEYS_FILE")) as infile:
            word_lists = yaml.load(infile, Loader=yaml.Loader)

        with open(config.get("SNORKEL_LABELLING", "MULTI_KEYS_FILE")) as infile:
            multi_word_lists = yaml.load(infile, Loader=yaml.Loader)
        multi_word_lists = {
            TOPICS[topic]: words for topic, words in multi_word_lists.items()
        }
        word_lists = {TOPICS[topic]: words for topic, words in word_lists.items()}

        full_df = pd.read_csv(config.get("INPUTS", "SNORKEL_TRAINING_DATA"))
        full_df = full_df.loc[full_df[text_column].notnull()]  # remove nulls
        print(f"number of Full DF texts: {len(full_df)}")

        annotated_df = pd.read_csv(config.get("INPUTS", "ANNOTATED_DATA"))
        annotated_df = annotated_df.loc[
            annotated_df[text_column].notnull()
        ]  # remove nulls
        print(f"number of annotated texts: {len(annotated_df)}")

    # standardize the text

    with PrintContext("Standardizing Text"):
        annotated_df["standardized_text"] = annotated_df[text_column].progress_apply(
            lambda x: standardize(x)
        )
        full_df["standardized_text"] = full_df[text_column].progress_apply(
            lambda x: standardize(x)
        )

    print(annotated_df.keys())
    annotated_df["label_value"] = annotated_df[label_column].apply(
        lambda x: TOPICS[x.strip()].value
        if x != "" and x != None
        else TOPICS.UNDECIDABLE
    )

    # create all the functions
    LFS = []
    for topic in word_lists.keys():
        for keyword in word_lists[topic]:
            LFS += [
                fn_from_topic_and_keyword(topic, keyword, True, True),
                fn_from_topic_and_keyword(topic, keyword, False, True),
                fn_from_topic_and_keyword(topic, keyword, True, False),
                fn_from_topic_and_keyword(topic, keyword, False, False),
            ]

    for topic in multi_word_lists.keys():
        for words in multi_word_lists[topic]:
            LFS += [
                fn_from_topic_and_wordlist(topic, words, False),
                fn_from_topic_and_wordlist(topic, words, True),
            ]
    lengths = [5, 7, 9, 11, 13]
    for length in lengths:
        LFS += [
            # undecidable_length(length, True),
            undecidable_length(length, False)
        ]

    # create the label matrices, a matrix of how each lf votes
    applier = PandasLFApplier(lfs=LFS)
    training_label_matrix = applier.apply(df=full_df)

    full_analysis = LFAnalysis(L=training_label_matrix, lfs=LFS)
    full_analysis.lf_summary()

    # remove all the lfs that never vote, and redo the analysis
    print(f"Initial number of LFS:{len(LFS)}")
    cov_analysis = full_analysis.lf_coverages()
    lost_lfs = [LF for LF, cov in zip(LFS, cov_analysis) if cov == 0.0]
    LFS = [LF for LF, cov in zip(LFS, cov_analysis) if cov > 0]
    print(f"LFS Remaining:{len(LFS)}")
    #  print("DELETED LFS")
    #  for lf in lost_lfs:
    #      print(lf)

    # we apply our remaining lfs to the annotated set
    applier = PandasLFApplier(lfs=LFS)
    testing_label_matrix = applier.apply(df=annotated_df)
    annotated_analysis = LFAnalysis(L=testing_label_matrix, lfs=LFS)
    annotated_summary = annotated_analysis.lf_summary(annotated_df.label_value.values)

    # collect the LFS that are accurate according to our validation
    # remove any lf with less than 50% accuracy
    # we can only do this for the labelled set, so its a bit odd that way.
    accurateLFS = [
        LF for LF, acc in zip(LFS, annotated_summary["Emp. Acc."]) if acc >= 0.5
    ]

    # Collect the LFS that do no harm on our validation
    agreeableLFS = [
        LF for LF, confl in zip(LFS, annotated_summary["Conflicts"]) if confl < 0.0001
    ]

    # change our LFS
    # NEW_LFS = accurateLFS
    NEW_LFS = list(set(accurateLFS + agreeableLFS))
    print(f"LFS Remaining:{len(NEW_LFS)}")
    print(f"Agreeable count: {len(agreeableLFS)} Accurate Count: {len(accurateLFS)}")

    applier = PandasLFApplier(lfs=NEW_LFS)
    training_label_matrix = applier.apply(df=full_df)
    analysis = LFAnalysis(L=training_label_matrix, lfs=NEW_LFS)
    testing_label_matrix = applier.apply(df=annotated_df)
    summary = analysis.lf_summary()

    # Check to see if there is representation for every topic
    topic_ids = {t.value for t in TOPICS}
    topic_ids_used = set(z for y in summary["Polarity"] for z in y)
    topic_ids_used.add(-1)  # abstain
    if topic_ids != topic_ids_used:
        print(f"MISSING:{topic_ids - topic_ids_used}")
    summary

    majority_model = MajorityLabelVoter(cardinality=len(TOPICS), verbose=True)
    preds_train = majority_model.predict(L=training_label_matrix)

    majority_acc = majority_model.score(
        L=testing_label_matrix,
        Y=annotated_df.label_value.values,
        tie_break_policy="random",
    )["accuracy"]
    print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

    label_model = majority_model
    # LabelModel(cardinality=len(TOPICS), verbose=True)

    snorkel_model_path = "models/snorkel_label_model"
    # label_model.load(snorkel_model_path)

    # print("Training model")
    # label_model.fit(training_label_matrix, n_epochs=int(config.get("SNORKEL_LABELLING", "EPOCHS")), seed=42)

    # label_model_acc = label_model.score(
    #     L=testing_label_matrix,
    #     Y=annotated_df.label_value.values,
    #     tie_break_policy="random",
    # )["accuracy"]
    # print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
    # #free some memory
    # training_label_matrix = ""
    # analysis = ""
    # testing_label_matrix =""
    # summary = ""
    # majority_model = ""
    # preds_train=""

    print("Labelling target data...")

    for p in SNORKEL_to_label_paths:

        df_to_label = pd.read_csv(p)
        df_to_label = df_to_label[df_to_label[text_column].notna()]
        print(f"Number of entries to label: {len(df_to_label)}")

        label_df(df_to_label, applier, label_model, text_column)

        df_to_label.to_csv(p + "_snorkel_labelled.csv")

    print("Labelling training set")
    full_df = full_df[full_df[text_column].notna()]
    label_df(full_df, applier, label_model, text_column, standardize=False)
    full_df.to_csv(config.get("OUTPUT", "SNORKEL_FULL_LABELLED_OUT"))
