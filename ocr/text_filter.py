"""
ocr/text_filter.py - turn raw OCR words into something worth saying (or nothing).

Three jobs:
  1. CONFIDENCE  Tesseract gives every word a confidence 0-100. Words below
                 min_word_conf are dropped: on a camera image, low-confidence
                 "words" are mostly textures (wood grain, fabric) read as letters.
  2. JUNK        tokens without any letter or digit ("|", "~~", "—") are dropped.
                 Short tokens (1-2 letters) are only kept on a line that also has
                 a real word of 3+ letters or a 2+ digit number: "Take 1 a day" is kept,
                 but a lone "ay" or "ie" (what Tesseract makes of shadows and
                 patterns, found in testing) is dropped.
  3. REPEATS     in continuous mode the camera sees the same sign every 2 s.
                 Two texts count as the same when difflib's similarity ratio
                 (Ratcliff/Obershelp: 2*matches / total length) is above
                 duplicate_similarity, so one misread letter does not make it
                 "new". The same text is not spoken again within repeat_cooldown_s.
"""

import difflib
import re


def _core(token):
    return re.sub(r"[^0-9A-Za-z]", "", token)


def is_junk(token):
    """No plain letter or digit at all (also catches stray accents like 'é')."""
    return not _core(token)


def is_strong(token):
    """A word that can stand on its own: 3+ letters/digits, or a 2+ digit number."""
    core = _core(token)
    return len(core) >= 3 or (core.isdigit() and len(core) >= 2)


def confident_words(words, min_conf):
    kept = [w for w in words if w.conf >= min_conf and not is_junk(w.text)]
    strong_lines = {(w.block, w.par, w.line) for w in kept if is_strong(w.text)}
    return [w for w in kept if (w.block, w.par, w.line) in strong_lines]


def words_to_text(words):
    """Keep Tesseract's reading order: block -> paragraph -> line -> word."""
    ordered = sorted(words, key=lambda w: (w.block, w.par, w.line, w.word_num))
    lines, current_key, current = [], None, []
    for w in ordered:
        key = (w.block, w.par, w.line)
        if key != current_key and current:
            lines.append(" ".join(current))
            current = []
        current_key = key
        current.append(w.text)
    if current:
        lines.append(" ".join(current))
    return ". ".join(lines)


def normalise(text):
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z ]", " ", text.lower())).strip()


def similarity(a, b):
    return difflib.SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def truncate_words(text, max_words):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ... and more text"


def extract_text(words, config):
    """Raw Tesseract words -> clean text, or "" when nothing trustworthy is left."""
    kept = confident_words(words, config.min_word_conf)
    text = words_to_text(kept)
    if sum(ch.isalnum() for ch in text) < config.min_text_chars:
        return ""
    return truncate_words(text, config.max_spoken_words)


def is_unclear(words, config):
    """True when Tesseract itself was unsure: a word we would speak scored below
    sure_word_conf, or a real-looking word was dropped for low confidence (so the
    spoken text has a gap). SENSE then says the reading may be wrong, because a
    misread medicine dose must never sound certain."""
    kept = confident_words(words, config.min_word_conf)
    if any(w.conf < config.sure_word_conf for w in kept):
        return True
    return any(config.unclear_floor_conf <= w.conf < config.min_word_conf and is_strong(w.text)
               for w in words)


class RepeatFilter:
    """Remembers what was spoken recently, to skip near-identical repeats."""

    def __init__(self, similarity_threshold, cooldown_s):
        self.similarity_threshold = similarity_threshold
        self.cooldown_s = cooldown_s
        self.recent = []            # [(time_spoken, text)]

    def _prune(self, now):
        self.recent = [(t, s) for t, s in self.recent if now - t < self.cooldown_s]

    def is_repeat(self, text, now):
        self._prune(now)
        return any(similarity(text, s) >= self.similarity_threshold for _, s in self.recent)

    def remember(self, text, now):
        self._prune(now)
        self.recent.append((now, text))

    def clear(self):
        self.recent = []
