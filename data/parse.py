import re

parens = re.compile(r"^\(.*\)")
headingExtractor = re.compile(r"\s.*?(?=\s{2})")
breakLocator = re.compile(r"\s{2,}")
# isMain = re.compile(r'^[A-Z]
# def classify(heading):
#     if heading


def getFirstBreak(text):
    m = breakLocator.match(text)
    print("nein")
    if m:
        return m.start(0)
    return 0


def getHeading(text):
    m = headingExtractor.match(text)
    if m:
        return m.group(0)
    return ""


def readText(fileName="electionActLabeled.txt"):
    with open(fileName) as f:
        lines = [line.strip() for line in f.readlines()]
        headings = [line[getFirstBreak(line)] for line in lines]
    return lines, headings


def isTitleHeading(text):
    return len(text) > 1 and text[0].isnumeric()


def getInParens(text):
    m = parens.match(text)
    if m:
        return m.group(0)[1:-1]
    else:
        return m
