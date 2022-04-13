import re
import string
import math
import numpy as np

from functools import lru_cache

from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import spacy

# Spacy setup

# this method is adapted from https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
table = str.maketrans({key: None for key in string.punctuation + "’" + "“" + "”"})
seed_finder = re.compile(r"“.+”")  # note that this wont work if punctuation is removed

nlp = spacy.load("en_core_web_lg")
nlp.tokenizer.rules = {
    key: value for key, value in nlp.tokenizer.rules.items() if "cannot" not in key
}


def spacyize(text):
    spac = nlp(text)
    return {
        "lemmas": [token.lemma_ for token in spac],
        "pos": [token.pos_ for token in spac],
        "stops": [token.is_stop for token in spac],
        "tokens": [token.text for token in spac],
    }


# custom punctuation stripper
punct = set(string.punctuation + "’" + "“" + "”")
punct.remove(")")
punct.remove("(")


# embedding model
def remove_punctuation(text):
    return text.translate(table)


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print("module %s loaded" % module_url)


def embed(input):
    return model(input)


def get_seeds(text):
    m = seed_finder.match(text)
    if m:
        return m.group(0).strip("“”")


# seeding methods / evaluation
eval_data = None


def set_eval_data(eval_data_set):
    global eval_data
    eval_data = eval_data_set


def evaluate_entity_sets(entity_sets):
    results = []
    total_correctly_matched = 0
    total_false_positives = 0
    total_false_negatives = 0

    for s, v in zip(entity_sets, eval_data.entity_set):
        correctly_matched = len(s.intersection(v))
        false_positive = len(s - v)
        false_negative = len(v - s)
        total_correctly_matched += correctly_matched
        total_false_positives += false_positive
        total_false_negatives += false_negative
        results.append(
            {
                "correct": correctly_matched,
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        )
    precision = total_correctly_matched / (
        total_correctly_matched + total_false_positives
    )
    recall = total_correctly_matched / (total_correctly_matched + total_false_negatives)
    f1 = 2 * ((precision * recall) / (precision + recall))
    print(f"Entity Set Precision: {precision}")
    print(f"Entity Set Recall: {recall}")
    print(f"Entity Set F1: {f1}")
    return {"precision": precision, "recall": recall, "f1": f1, "results": results}


def evaluate_entity_token_sets(entity_sets):
    results = []
    total_correctly_matched = 0
    total_false_positives = 0
    total_false_negatives = 0

    for s, v in zip(entity_sets, eval_data.entity_token_set):
        correctly_matched = len(s.intersection(v))
        false_positive = len(s - v)
        false_negative = len(v - s)
        total_correctly_matched += correctly_matched
        total_false_positives += false_positive
        total_false_negatives += false_negative
        results.append(
            {
                "correct": correctly_matched,
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        )
    precision = total_correctly_matched / (
        total_correctly_matched + total_false_positives
    )
    recall = total_correctly_matched / (total_correctly_matched + total_false_negatives)
    f1 = 2 * ((precision * recall) / (precision + recall))
    print(f"Entity Token Set Precision: {precision}")
    print(f"Entity Token Set Recall: {recall}")
    print(f"Entity Token Set F1: {f1}")
    return {"precision": precision, "recall": recall, "f1": f1, "results": results}


class Span:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.len = end - start

    def __len__(self):
        return self.len

    def overlap(self, other):
        return max(self.start, other.start) <= min(self.end, other.end)

    def overlaps(self, other):
        return self.overlap(other)

    def __repr__(self):
        return f"start: {self.start} end: {self.end}"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__repr__())


def char_spans_to_token_span(char_spans, text):
    tokens = tokenize(text)

    token_spans = []
    # print(char_spans)

    for span in char_spans:
        this_span_start = -1
        this_span_end = -1
        start = 0
        end = 0
        for i, t in enumerate(tokens):
            end += len(t)
            t_span = Span(start, end)
            start = end
            overlaps = span.overlap(t_span)

            if overlaps and this_span_start == -1 and t != " ":
                this_span_start = i
            elif overlaps:
                this_span_end = i
            elif this_span_start != -1 and this_span_end != -1:
                token_spans.append(Span(this_span_start, this_span_end))
                break
        else:
            if this_span_start == -1 or this_span_end == -1:
                return []
            token_spans.append(Span(this_span_start, len(tokens)))

    return token_spans


class ReSearch:
    def __init__(self, text_re, pos_re):
        self.text_re = text_re
        self.pos_re = pos_re

    def match_text(self, text):
        spans = []
        for m in self.text_re.finditer(text):
            s = Span(m.start(), m.end())
            if any([s.overlap(x) for x in spans]):
                continue
            spans.append(s)
        return char_spans_to_token_span(spans, text)

    def match_pos(self, poses):
        if isinstance(poses, list):
            poses = " ".join(str(x) for x in poses)
        return char_spans_to_token_span(
            [Span(m.start(), m.end()) for m in self.pos_re.finditer(poses)], poses
        )

    def match_both(self, text, poses):

        text_spans = set(self.match_text(text))
        pos_spans = set(self.match_pos(poses))
        return set(text_spans).intersection(set(pos_spans))


class Seed(ReSearch):
    def __init__(self, text):
        # handle text
        self.text = remove_punctuation(text.lower())
        spac = spacyize(self.text)
        self.lemmas = spac["lemmas"]
        self.pos = spac["pos"]
        self.stops = spac["stops"]
        self.embedding = embed([self.text])

        # handle pos tags
        self.tags = str(" ".join(x for x in self.pos))
        super().__init__(
            re.compile(f"\\b{self.text}\\b", re.I),
            re.compile(f"\\b{self.tags}\\b", re.I),
        )

    def __repr__(self):
        return f'Text: "{self.text}"' + "\n" + f'Pos: "{self.tags}"'

    def __str__(self):
        return self.__repr__()

    def fullMatch(self, text, tags):
        return self.match_both(text, tags)


@lru_cache()
def get_pos_tags(text):
    spac = spacyize(text)
    return " ".join(str(x) for x in spac["pos"])


@lru_cache()
def tokenize(text):
    return re.split("(\W)", text)


class SeedList:
    def __init__(self, startList):
        self.seeds = [Seed(text) for text in startList]
        self.sort()

    def add(self, text, sort=True):
        self.seeds.append(Seed(text))
        if sort:
            self.sort()

    def sort(self):
        self.seeds.sort(key=lambda x: len(x.text), reverse=True)

    def __len__(self):
        return len(self.seeds)

    def get_spans_from_seeds(self, text):
        spans = set()
        tokens = tokenize(text)
        for i, seed in enumerate(self.seeds):
            matches = seed.match_text(text)
            for match in matches:
                for span in spans:
                    if span.overlaps(match):
                        break
                else:
                    spans.add(match)
        return spans

    def get_entity_sets_from_seeds(self, text, whole=True, both=False):
        entities = set()
        spans = set()
        tags = get_pos_tags(text)
        tokens = tokenize(text)
        for i, seed in enumerate(self.seeds):
            if both:
                matches = seed.fullMatch(text, tags)
            else:
                matches = seed.match_text(text)
            for match in matches:
                for span in spans:
                    if span.overlaps(match):
                        break
                else:
                    spans.add(match)
                    if whole:
                        entities.add("".join(tokens[match.start : match.end]))
                    else:
                        entities.update(
                            t for t in tokens[match.start : match.end] if t != " "
                        )

        return entities

    def score(self, text):
        emb = embed([text])
        sc = (0.0, "")
        for seed in self.seeds:
            this_score = np.inner(seed.embedding, emb)
            if this_score > sc[0]:
                sc = (float(this_score), seed.text)
        return sc


class Pattern(ReSearch):
    def __init__(self, prior_pos, entity_pos, post_pos, entity_seed):
        self.prior_pos = prior_pos
        self.entity_pos = entity_pos
        self.post_pos = post_pos
        self.pattern = prior_pos + entity_pos + post_pos
        self.prior_len = len(prior_pos) * 2 - 1  # account for the spaces....
        self.post_len = len(post_pos) * 2 - 1  # account for the spaces....
        self.entity_len = len(entity_pos)
        self.len = len(self.pattern)
        self.entity_seed = entity_seed
        self.pattern_text = " ".join(x for x in self.pattern)
        self.entities = set()
        super().__init__(
            re.compile(f"\\b{self.entity_seed}\\b", re.I),
            re.compile(f"\\b{self.pattern_text}\\b", re.I),
        )

    def __repr__(self):
        return ", ".join(self.pattern)

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self.len

    def __eq__(self, other):
        if isinstance(other, Pattern):
            return self.__repr__() == other.__repr__()
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__repr__())

    def get_entities(self, text, tag_string, both=False):
        if isinstance(text, str):
            tokens = tokenize(text)
        if isinstance(text, list):
            tokens = text
            text = "".join(tokens)

        if not isinstance(tag_string, str):
            tag_string = " ".join(tag_string)

        entity_matches = set()

        if both:
            pos_spans = self.match_both(text, tag_string)
        else:
            pos_spans = self.match_pos(tag_string)
        for span in pos_spans:
            entity_matches.add(
                "".join(
                    tokens[span.start + self.prior_len : span.end - self.post_len]
                ).strip()
            )
        return entity_matches


class Patterns:
    def __init__(self, startList):
        self.seedList = SeedList(startList)
        self.eval_data = eval_data
        self.seed_set = startList
        self.results = []
        self.patterns = set()

    def print_seeds(self):
        for x in self.seedList.seeds:
            print(x)

    def __len__(self):
        return len(self.seedList)

    def check_seeds(self, text, whole=True):
        return self.seedList.get_entity_sets_from_seeds(text, whole=whole)

    def add_seed(self, text):
        self.seedList.add(text)
        self.seed_set.add(text)

    def score(self, text):
        return self.seedList.score(text)

    def evaluate(self, new_candidates=[]):
        initial_seed_set = self.eval_data.text.apply(self.check_seeds, whole=True)
        r = evaluate_entity_sets(initial_seed_set)

        initial_token_set = self.eval_data.text.apply(self.check_seeds, whole=False)
        rtok = evaluate_entity_token_sets(initial_token_set)

        self.results.append(
            {
                "F1": r["f1"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "Partial F1": rtok["f1"],
                "Partial Recall": rtok["recall"],
                "Partial Precision": rtok["precision"],
                "New Candidates": new_candidates,
            }
        )

        if len(self.results) > 1:
            for x in {
                "F1",
                "Precision",
                "Recall",
                "Partial F1",
                "Partial Recall",
                "Partial Precision",
            }:
                print(
                    f"{x:>20} changed from {self.results[-2][x]:.2f} to {self.results[-1][x]:.2f}, a change of {self.results[-1][x]-self.results[-2][x]:.2f}"
                )

        return r, rtok

    def get_spans(self, text):
        return self.seedList.get_spans_from_seeds(text)

    def build_patterns(self, train_data, pre_window=4, post_window=4):

        for a, row in train_data.iterrows():
            spans = self.get_spans(row.text)
            tokens = tokenize(row.text)
            pos = row.pos
            for span in spans:
                pos_start = span.start - span.start // 2
                pos_end = span.end - span.end // 2
                for a in range(2, pre_window + 1):
                    for b in range(2, post_window + 1):
                        p = Pattern(
                            pos[max(0, pos_start - a) : pos_start],
                            pos[pos_start:pos_end],
                            pos[pos_end : pos_end + b],
                            "".join(tokens[span.start : span.end]),
                        )
                        if p not in self.patterns:
                            self.patterns.add(p)

        print(len(self.patterns))

    def run(self, df):
        for pattern in tqdm(self.patterns):
            pattern_entities = set()
            if len(pattern.entities) == 0:
                for _, row in df.iterrows():
                    pattern_entities.update(pattern.get_entities(row.text, row.pos))
                pattern.entities = pattern_entities

            pattern.positives = self.seed_set.intersection(pattern.entities)
            pattern.negatives = pattern.entities - self.seed_set
            if len(pattern.positives) == 0 or len(pattern.negatives) == 0:
                pattern.overall = 0
                pattern.score = 0
                continue

            pattern.score = sum([1 - self.score(x)[0] for x in pattern.negatives])
            pattern.overall = (
                len(pattern.positives)
                / (len(pattern.negatives) + pattern.score + 0.00000000000001)
                * math.log(len(pattern.positives))
            )

        ranked = sorted(list(self.patterns), reverse=True, key=lambda x: x.overall)
        new_candidates = set()
        for r in ranked[:50]:
            new_candidates.update(r.negatives)
        new_candidates = sorted(
            new_candidates, key=lambda x: self.score(x), reverse=True
        )
        for cand in new_candidates[:10]:
            self.add_seed(cand)
        self.evaluate(new_candidates[:10])
        print(self.results[-1])


# self references
number_finder = re.compile(
    r"((?:sections?)|(?:parts?)|(?:division)|(?:subsections?)|(?:subsubsections?)){0,1} {0,1}(?:(\d+\.*\d*)|(\(\d+\))|(\([a-z]+\))+)+"
)


def find_possible_referrals(text):
    matches = number_finder.finditer(text, re.MULTILINE)
    return [str(match.group()).strip() for match in matches]


def find_referrals_in_df(df, G):
    df["self_references"] = df.node.apply(
        lambda x: find_possible_referrals(G.nodes()[x]["text"])
    )


candidates = None


def set_candidates(G):
    global candidates
    candidates = sorted(G.nodes(), key=lambda x: len(x), reverse=True)[
        :-1
    ]  # trim the root, we never want the root


class Target:
    def __init__(self, s, sourceNode):
        if "subsection" in s:
            self.level = 8
            self.part = sourceNode["part"]
            self.division = sourceNode["division"]
            self.section = sourceNode["section"]
            self.subsection = "0"  # s.split()[-1].replace('(', '').replace(')','')
            self.subsubsection = "0"
            self.subsubsubsection = "0"
            nums = s.split()[-1]
            nums = [
                n.strip(")") for n in nums.split("(")
            ]  # we split on open parens, the base is the section

            self.subsection = nums.pop(0)
            self.subsection = nums.pop(0)
            if len(nums) > 0:
                self.subsubsection = nums.pop(0)
            if len(nums) > 0:
                self.subsubsubsection = nums.pop(0)
        elif "part" in s:
            self.level = 2
            self.part = s.split()[-1]
            self.division = "0"
            self.section = "0"
            self.subsection = "0"
            self.subsubsection = "0"
            self.subsubsubsection = "0"

        elif "division" in s:
            self.level = 4
            self.part = sourceNode["part"]
            self.division = s.split()[-1]
            self.section = "0"
            self.subsection = "0"
            self.subsubsection = "0"
            self.subsubsubsection = "0"
        elif "section" in s:
            self.level = 7  # default to section level
            self.part = sourceNode["part"]
            self.division = sourceNode["division"]
            self.section = "0"
            self.subsection = "1"
            self.subsubsection = "0"
            self.subsubsubsection = "0"

            nums = s.split()[-1]
            nums = [
                n.strip(")") for n in nums.split("(")
            ]  # we split on open parens, the base is the section
            self.section = nums.pop(0)
            if len(nums) > 0:
                self.subsection = nums.pop(0)
            if len(nums) > 0:
                self.subsubsection = nums.pop(0)
            if len(nums) > 0:
                self.subsubsubsection = nums.pop(0)
        else:
            if "(" not in s and "." not in s:
                raise ValueError("No entity detected")
            self.level = 7  # default to section level
            self.part = sourceNode["part"]
            self.division = sourceNode["division"]
            self.section = "0"
            self.subsection = "0"
            self.subsubsection = "0"
            self.subsubsubsection = "0"

            nums = s.split()[-1]
            nums = [
                n.strip(")") for n in nums.split("(")
            ]  # we split on open parens, the base is the section
            self.section = nums.pop(0)
            if len(nums) > 0:
                self.subsection = nums.pop(0)
            if len(nums) > 0:
                self.subsubsection = nums.pop(0)
            if len(nums) > 0:
                self.subsubsubsection = nums.pop(0)

    def __repr__(self):
        return f"""          level: {self.level}
                    part: {self.part}
                division: {self.division}
                 section: {self.section}
              subsection: {self.subsection}
           subsubsection: {self.subsubsection}
        subsubsubsection: {self.subsubsubsection}
        """

    def comp(self, a, b):
        # if a == 0: return True
        return a == b

    def matches(self, node):
        if self.level == 7:
            return (
                sum(
                    [
                        self.comp(self.section, node["section"]),
                        self.comp(self.subsection, node["subsection"]),
                        self.comp(self.subsubsection, node["subsubsection"]),
                        self.comp(self.subsubsubsection, node["subsubsubsection"]),
                    ]
                )
                == 4
            )
        return (
            sum(
                [
                    self.comp(self.part, node["part"]),
                    self.comp(self.division, node["division"]),
                    self.comp(self.section, node["section"]),
                    self.comp(self.subsection, node["subsection"]),
                    self.comp(self.subsubsection, node["subsubsection"]),
                    self.comp(self.subsubsubsection, node["subsubsubsection"]),
                ]
            )
            == 6
        )


def validate(nodeName, references, G):
    node = G.nodes(data=True)[nodeName]
    matches = []
    for ref in references:
        thisNode = node
        try:
            target = Target(ref, thisNode)
        except ValueError:
            continue

        matched = False

        for c in candidates:
            thisNode = G.nodes(data=True)[c]
            if "level" not in thisNode:
                continue
            if int(thisNode["level"]) < int(target.level):
                continue
            if target.matches(thisNode):
                print(f'Added Edge: FROM "{node["name"]}" TO "{thisNode["name"]}"')
                G.add_edge(node["name"], thisNode["name"])
                print(node["text"])
                print(f"by reference {ref}")
                print("=" * 80)
                matched = True
                break

        matches.append(matched)
        if not matched:
            print(f"FAILED to match {node}")
            print(target)

            print("=" * 80)
    return [r for r, m in zip(references, matches) if m]
