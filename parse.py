import re
import inspect
from itertools import islice

parens = re.compile(r"\(.*\)")
spaceFinder = re.compile(r"\s{2,}")
dateFinder = re.compile(r"\d{4}")


def genAlpha():
    alphas = "abcdefghijklmnopqrstuvwxyz"
    multi = 1
    while True:
        for letter in alphas:
            yield letter * multi
        multi += 1


letters = [*islice(genAlpha(), 500)]

itoa = {i: x for i, x in enumerate(letters)}
atoi = {x: i for i, x in enumerate(letters)}
itor = {
    i: x
    for i, x in enumerate(["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"])
}
rtoi = {
    x: i
    for i, x in enumerate(["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"])
}


def handleSubbing(heading):
    if "." in heading:
        return heading.split(".")
    return [heading]


def predictNextAlpha(heading):
    if heading not in atoi:
        return None
    if heading in atoi:
        return itoa[atoi[heading] + 1]


def predictNextRoman(heading):
    if heading not in rtoi:
        return None
    if heading in rtoi:
        return itor[rtoi[heading] + 1]


def predictPreviousAlpha(heading):
    if heading not in atoi:
        return None
    if heading == "a":
        return -1
    if heading in atoi:
        return itoa[atoi[heading] - 1]


def predictPreviousRoman(heading):
    if heading not in rtoi:
        return None
    if heading == "i":
        return -1
    if heading in rtoi:
        return itor[rtoi[heading] - 1]


def predictNextNum(heading):
    if not heading.isnumeric():
        return None
    return str(int(heading) + 1)


def getBreak(text):
    m = spaceFinder.search(text)

    if m:
        return m.start(0)
    return None


class headings:
    def __init__(self):
        self.lastAction = inspect.currentframe().f_code.co_name
        self.part = "0"
        self.partName = "Part"
        self.division = "0"
        self.divisionName = "Division"
        self.sectionName = "Section"
        self.section = "0"
        self.subsection = "0"
        self.subsubsection = "0"
        self.subsubsubsection = "0"
        self.subsubsubsubsection = "0"
        self.lastAction = None

    def setPart(self, part):
        self.lastAction = inspect.currentframe().f_code.co_name
        self.part = part
        self.division = "0"
        self.divisionName = "Division"
        self.sectionName = "Section"
        self.section = "0"
        self.subsection = "0"
        self.subsubsection = "0"
        self.subsubsubsection = "0"
        self.subsubsubsubsection = "0"

    def setPartName(self, partName):
        self.lastAction = inspect.currentframe().f_code.co_name
        self.partName = partName
        self.division = "0"
        self.divisionName = "Division"
        self.sectionName = "Section"
        self.section = "0"
        self.subsection = "0"
        self.subsubsection = "0"
        self.subsubsubsection = "0"
        self.subsubsubsubsection = "0"

    def setDivision(self, division):
        self.lastAction = inspect.currentframe().f_code.co_name
        self.division = division
        self.sectionName = "Section"
        self.section = "0"
        self.subsection = "0"
        self.subsubsection = "0"
        self.subsubsubsection = "0"
        self.subsubsubsubsection = "0"

    def setDivisionName(self, divisionName):
        self.lastAction = inspect.currentframe().f_code.co_name
        self.divisionName = divisionName
        self.sectionName = "Section"
        self.section = "0"
        self.subsection = "0"
        self.subsubsection = "0"
        self.subsubsubsection = "0"
        self.subsubsubsubsection = "0"

    def setSectionName(self, sectionName):
        self.lastAction = inspect.currentframe().f_code.co_name
        self.sectionName = sectionName
        self.subsection = "0"
        self.subsubsection = "0"
        self.subsubsubsection = "0"
        self.subsubsubsubsection = "0"

    def setSection(self, section):
        self.lastAction = inspect.currentframe().f_code.co_name
        self.section = section
        self.subsection = "0"
        self.subsubsection = "0"
        self.subsubsubsection = "0"
        self.subsubsubsubsection = "0"

    def setSubSection(self, index):
        self.lastAction = inspect.currentframe().f_code.co_name
        self.subsection = index
        self.subsubsection = "0"
        self.subsubsubsection = "0"
        self.subsubsubsubsection = "0"

    def setSubSubSection(self, index):
        self.lastAction = inspect.currentframe().f_code.co_name
        self.subsubsection = index
        self.subsubsubsection = "0"
        self.subsubsubsubsection = "0"

    def setSubSubSubSection(self, index):
        self.lastAction = inspect.currentframe().f_code.co_name
        self.subsubsubsection = index
        self.subsubsubsubsection = "0"

    def setSubSubSubSubSection(self, index):
        self.lastAction = inspect.currentframe().f_code.co_name
        self.subsubsubsubsection = index

    def tolist(self):
        return [
            self.partName,
            self.part,
            self.divisionName,
            self.division,
            self.sectionName,
            self.section,
            self.subsection,
            self.subsubsection,
            self.subsubsubsection,
            self.subsubsubsubsection,
        ]

    def __str__(self):
        return str(self.tolist())

    def __repr__(self):
        return self.__str__()


currentHeadings = headings()


def headingParser(text):
    global currentHeadings, partLast, divLast
    if text[0].isalpha():
        if currentHeadings.lastAction == "setPart":
            currentHeadings.setPartName(text)
            return
        elif currentHeadings.lastAction == "setDivision":
            currentHeadings.setDivisionName(text)
            return

        if text[:4] == "Part":  # the part is the most basic of headings
            _, num = text.split(" ")
            currentHeadings.setPart(num)
            partLast = True
            return
        else:
            partLast = False

        if text[:4] == "Divi":  # the part is the most basic of headings
            _, num = text.split(" ")
            currentHeadings.setDivision(num)
            divLast = True
            return
        else:
            divLast = False

        # we cant figure out what else it should be...
        currentHeadings.setSectionName(text)
        return

    if text[0].isnumeric():
        if "(" in text:
            main_section, sub_section = text.split("(")
            sub_section = sub_section[:-1]  # trim off the last bracket
            currentHeadings.setSection(main_section)
            currentHeadings.setSubSection(sub_section)
            return
        else:
            currentHeadings.setSection(text)
            return

    if text[0] == "(":
        text = text[1:-1]
        if text[0].isnumeric():
            currentHeadings.setSubSection(text)
            return
        if "." in text:
            a, b = text.split(".")
            if currentHeadings.subsubsection[: len(a)] == a:
                currentHeadings.setSubSubSection(text)
                return
            if currentHeadings.subsubsubsection[: len(a)] == a:
                currentHeadings.setSubSubSubSection(text)
                return
        else:
            prevAlpha = predictPreviousAlpha(text)
            if prevAlpha == -1:  # starting tag
                currentHeadings.setSubSubSection(text)
                return

            prevRoman = predictPreviousRoman(text)
            if (
                prevAlpha
                and prevAlpha
                == currentHeadings.subsubsection[: len(prevAlpha)]
                == prevAlpha
            ):
                currentHeadings.setSubSubSection(text)
                return

            elif prevRoman == -1 or (
                prevRoman
                and prevRoman
                == currentHeadings.subsubsubsection[: len(prevRoman)]
                == prevRoman
            ):

                currentHeadings.setSubSubSubSection(text)
                return
            if text in "ABCDEFGHIJKL":
                currentHeadings.setSubSubSubSubSection(text)
                return
    raise ValueError(f"Unable to classify: '{text}'")


def readText(fileName="electionActLabeled.txt"):
    with open(fileName) as f:
        lines = [line.strip() for line in f.readlines()]
        lines = [
            line
            for line in lines
            if len(line) > 0
            and line[:3] != "RSA"
            and not dateFinder.match(line)
            and "Repealed" not in line
        ]
        headings = [line[: getBreak(line)] for line in lines]
        assert len(lines) == len(headings)
    return lines, headings


def isTitleHeading(text):
    return len(text) > 1 and text[0].isnumeric()


def getInParens(text):
    m = parens.match(text)
    if m:
        return m.group(0)[1:-1]
    else:
        return m
