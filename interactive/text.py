"""
interactive/text.py - normalising what Whisper wrote, and the words SENSE knows.

Whisper writes "What's the colour of the bottel?" as often as "what is the
color of the bottle". Everything downstream works on one normalised form.
"""
import difflib
import re

COLOUR_WORDS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple",
                "white", "black", "gray")
COLOUR_ALIASES = {"grey": "gray", "pink": "red", "violet": "purple", "brown": "orange",
                  "silver": "gray", "navy": "blue", "teal": "cyan"}

# spoken word -> SENSE class name (the names sense_master uses after ALLOWED_ITEMS)
CLASS_WORDS = {
    "person": "person", "people": "person", "persons": "person", "man": "person",
    "woman": "person", "men": "person", "women": "person", "someone": "person",
    "anyone": "person", "anybody": "person", "somebody": "person", "humans": "person",
    "human": "person", "guy": "person", "guys": "person", "kid": "person", "kids": "person",
    "phone": "phone", "phones": "phone", "mobile": "phone", "cellphone": "phone",
    "remote": "remote", "remotes": "remote",
    "cup": "cup", "cups": "cup", "mug": "cup", "mugs": "cup",
    "bottle": "bottle", "bottles": "bottle", "bottel": "bottle",
    "book": "book", "books": "book",
    "laptop": "laptop", "laptops": "laptop", "computer": "laptop",
    "mouse": "mouse", "mice": "mouse",
    "chair": "chair", "chairs": "chair", "seat": "chair", "seats": "chair",
    "couch": "couch", "couches": "couch", "sofa": "couch", "sofas": "couch",
    "bed": "bed", "beds": "bed",
    "table": "table", "tables": "table", "desk": "table",
    "plant": "plant", "plants": "plant",
    "tv": "tv", "tvs": "tv", "television": "tv",
    "dog": "dog", "dogs": "dog", "cat": "cat", "cats": "cat",
    "suitcase": "suitcase", "suitcases": "suitcase", "bag": "suitcase",
    "bench": "bench", "benches": "bench",
    "bicycle": "bicycle", "bicycles": "bicycle", "bike": "bicycle", "bikes": "bicycle",
    "door": "door", "doors": "door", "doorway": "door",
    "stairs": "stairs", "stair": "stairs", "staircase": "stairs", "steps": "stairs",
}
# what the detectors can actually see (anything else is an honest "I can't detect")
DETECTABLE = {"person", "phone", "remote", "cup", "bottle", "book", "laptop", "mouse",
              "chair", "couch", "bed", "table", "plant", "tv", "dog", "cat",
              "suitcase", "bench", "bicycle", "door", "stairs"}
SCENE_CLASSES = {"door", "stairs"}       # come from the second (Open Images) model
GENERIC_THING = {"object", "objects", "thing", "things", "item", "items", "stuff"}

# Common things people ask about that neither model can detect.
UNDETECTABLE_WORDS = {"key", "keys", "wallet", "wallets", "glasses", "spectacles", "pen",
                      "pens", "watch", "shoe", "shoes", "medicine", "pill", "pills",
                      "umbrella", "charger", "headphones", "earphones", "spoon", "fork",
                      "knife", "plate", "plates", "window", "windows", "car", "cars"}

_REPLACE = [
    (r"\bcolor(s|ed)?\b", r"colour\1"), (r"\bwhat's\b", "what is"), (r"\bwhere's\b", "where is"),
    (r"\bwho's\b", "who is"), (r"\bthere's\b", "there is"), (r"\bit's\b", "it is"),
    (r"\bhow's\b", "how is"), (r"\bi'm\b", "i am"), (r"\bthat's\b", "that is"),
    (r"\btoday's\b", "todays"), (r"\bwhats\b", "what is"), (r"\bwheres\b", "where is"),
    (r"\bin front\b", "in front"), (r"\bright hand side\b", "right"), (r"\bleft hand side\b", "left"),
    (r"\bright-hand side\b", "right"), (r"\bleft-hand side\b", "left"),
]


def normalise(text):
    t = (text or "").lower().replace("’", "'").replace("-", " ")
    for pat, rep in _REPLACE:
        t = re.sub(pat, rep, t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def words(text):
    return normalise(text).split()


_NOT_CLASS = {"there", "three", "these", "those", "where", "chain", "chat", "cap", "stars", "stays",
              "table's", "people's", "cups", "pairs", "chairs"}


def class_in(text):
    """First detectable class named in the text ('how many chairs' -> 'chair').
    Whisper misspellings ('bottels', 'chiars') are matched with difflib, the
    same Ratcliff/Obershelp similarity answers.fuzzy_find uses for names."""
    ws = words(text)
    for w in ws:
        if w in CLASS_WORDS:
            return CLASS_WORDS[w]
    for w in ws:
        if len(w) >= 5 and w not in _NOT_CLASS:
            m = difflib.get_close_matches(w, list(CLASS_WORDS), n=1, cutoff=0.8)
            if m:
                return CLASS_WORDS[m[0]]
    return None


def undetectable_in(text):
    for w in words(text):
        if w in UNDETECTABLE_WORDS:
            return w
    return None


def colour_in(text):
    for w in words(text):
        if w in COLOUR_WORDS:
            return w
        if w in COLOUR_ALIASES:
            return COLOUR_ALIASES[w]
    return None


def plural(cls, n):
    if n == 1:
        return cls
    irregular = {"person": "people", "mouse": "mice", "couch": "couches", "bench": "benches",
                 "stairs": "stairs", "tv": "TVs"}
    return irregular.get(cls, cls + "s")


def article(name):
    return "an" if name[:1] in "aeiou" else "a"


def number_word(n):
    return ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten"][n] if 0 <= n <= 10 else str(n)


def join_and(parts):
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return "".join(parts)
    if any("," in p for p in parts):       # "Sneha, 6 feet away, and a chair"
        return ", ".join(parts[:-1]) + ", and " + parts[-1]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def sentence(s):
    s = s.strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    return s if s[-1] in ".?!" else s + "."
