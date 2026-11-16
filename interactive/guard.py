"""
interactive/guard.py - fact-checking the local AI model before SENSE speaks.

Small language models are fluent but can "hallucinate": state things that
are not in their input. For a blind user a confident wrong answer is worse
than "I'm not sure", so every model answer is checked against the fact sheet
it was given (a simple, explainable form of "grounding" check):

  * every object word must be in the facts      ("the cup" when no cup -> reject)
  * every colour must be in the facts           ("a red bottle" -> reject)
  * every number must be in the facts           ("3 chairs" when facts say 2 -> reject)
  * every capitalised name must be in the facts ("Rahul is here" -> reject)
  * no safety promises                          ("the path is clear" -> reject)

A rejected answer is replaced by an honest fallback. This is stricter than it
needs to be on purpose: false alarms cost a re-ask, false facts cost trust.
"""
import re

from . import text as T

NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
SAFETY = re.compile(r"\b(safe|safely|clear path|path is clear|way is clear|is clear|no obstacles?|"
                    r"nothing in (your|the) way|go ahead|you can walk)\b", re.I)
ALLOWED_CAPS = {"I", "SENSE", "AM", "PM", "OK", "TV", "Sorry", "Yes", "No", "Monday", "Tuesday",
                "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February",
                "March", "April", "May", "June", "July", "August", "September", "October",
                "November", "December"}
# Ordinary words a sentence may start with. A capitalised word that is neither
# one of these nor in the facts is treated as an invented name ("Rahul is here").
STARTERS = set("""i the there it its you your yes no sorry a an this that these those they them he she
we on in to from at when earlier right now currently nothing nobody no one two three four five some
someone something anything everything all both only just also but and or so if not however please
here where what who which how why last before after today it's i'm i'd i've there's that's let let's
maybe probably perhaps unfortunately well okay ok hmm about around near next behind ahead straight
left turn walk go look ask try with without as for of by up down over under between inside outside
my mine our their his her yes my nothing""".split())
KNOWN_NAME_HINTS = set()   # filled by qa.py with every name SENSE has ever learned
PERSON_WORDS = {w for w, c in T.CLASS_WORDS.items() if c == "person"}


def clean(answer, max_sentences=2):
    a = (answer or "").replace("*", "").replace("#", "").replace("\n", " ").strip().strip('"')
    a = re.sub(r"\s+", " ", a)
    parts = re.split(r"(?<=[.!?])\s+", a)
    return " ".join(parts[:max_sentences]).strip()


def check(answer, facts):
    """(ok, reason). ok=False means: do not speak this answer."""
    if not answer or len(answer) < 2:
        return False, "empty"
    f = facts.lower()
    fw = set(re.findall(r"[a-z]+", f))
    words = re.findall(r"[A-Za-z]+", answer)
    lw = [w.lower() for w in words]
    if SAFETY.search(answer):
        return False, "safety promise"
    for w in lw:
        if w in T.CLASS_WORDS and w not in PERSON_WORDS:
            cls = T.CLASS_WORDS[w]
            if w not in fw and cls not in fw and cls + "s" not in fw:
                return False, f"object not in facts: {w}"
        if w in T.UNDETECTABLE_WORDS and w not in fw:
            # allowed only in "I can't detect keys" style sentences
            if not re.search(r"\b(can't|cannot|can not|don't|do not|unable|not able)\b", answer, re.I):
                return False, f"undetectable object claimed: {w}"
        if (w in T.COLOUR_WORDS or w in T.COLOUR_ALIASES) and w not in fw:
            return False, f"colour not in facts: {w}"
        if w in NUMBER_WORDS and str(NUMBER_WORDS[w]) not in f and w not in fw:
            return False, f"number not in facts: {w}"
    counts = _counts(f)
    for m in re.finditer(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\s+([a-z]+)(?:\s+([a-z]+))?", answer.lower()):
        n = int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS[m.group(1)]
        noun = m.group(2)
        if noun not in T.CLASS_WORDS and noun not in ("things", "objects", "items") and m.group(3):
            noun = m.group(3)            # "two black chairs", "3 big things"
        cls = T.CLASS_WORDS.get(noun)
        if noun in ("things", "objects", "items"):
            if n != sum(counts.values()):
                return False, f"wrong total: {n}"
        elif cls is not None and cls != "person" and counts.get(cls, 0) != n:
            return False, f"wrong count: {n} {noun}"      # counts are only known for what is in view
        elif cls == "person" and noun != "minutes" and counts.get("person", 0) != n:
            return False, f"wrong count: {n} {noun}"
    for num in re.findall(r"\d+", answer):
        if not re.search(rf"(?<!\d){num}(?!\d)", f):
            return False, f"number not in facts: {num}"
    for m in re.finditer(r"[A-Za-z]+", answer):
        w = m.group(0)
        starts_sentence = re.fullmatch(r"\s*|.*[.!?:;]\s*", answer[:m.start()]) is not None
        if w[0].isupper() and w not in ALLOWED_CAPS and w.lower() not in fw \
                and not (starts_sentence and w.lower() in STARTERS):
            return False, f"name not in facts: {w}"
    for name in re.findall(r"\b([A-Z][a-z]+)\b", answer):
        # a sentence-initial name ("Rahul is here.") still has to be a known word
        if name.lower() not in fw and name not in ALLOWED_CAPS and name.lower() in KNOWN_NAME_HINTS:
            return False, f"name not in facts: {name}"
    return True, "ok"


def _counts(facts_lower):
    m = re.search(r"counts in view: (.*)", facts_lower)
    out = {}
    if m:
        for part in m.group(1).split(","):
            bits = part.strip().rsplit(" ", 1)
            if len(bits) == 2 and bits[1].isdigit():
                out[bits[0]] = int(bits[1])
    return out


FALLBACK = ("I'm not sure about that. You can ask me what I see, where something is, "
            "or how many things are around you.")
