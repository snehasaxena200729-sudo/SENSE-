"""
ocr/quality.py - how far can we trust one Tesseract reading of a crop?

Tesseract's own word confidence (0-100) tells us how sure it is about each word
it DID read. It cannot tell us what it MISSED. The text detector can: it boxed
N words, so if Tesseract confidently read only a few of them, the reading is
incomplete even if every word it returned looks confident. So we combine:

  conf      character-weighted mean confidence of the words we would speak
  coverage  share of detector word boxes that contain a confidently read word
  cut       the block touches the frame edge, so text probably continues
            outside the picture

and label the reading:

  unreadable  nothing trustworthy -> never narrated, SENSE gives a camera tip
  partial     coverage below min_coverage or text cut at the edge
              -> "I can only read part of it: ..." + a tip
  unclear     everything covered, but a spoken word scored below sure_word_conf
              -> read with a warning (numbers named explicitly)
  clear       -> read normally

score (0..1) ranks several attempts of the same crop (different recipes):
confident characters / all characters Tesseract thought it saw, scaled by
coverage. A recipe that reads more of the text, more confidently, wins.
"""

import re
from collections import namedtuple

from .ocr_engine import Word
from .text_filter import confident_words, extract_text, is_junk, is_unclear

Reading = namedtuple("Reading", "text status score conf coverage cut unsure_numbers words")

CLEAR, UNCLEAR, PARTIAL, UNREADABLE = "clear", "unclear", "partial", "unreadable"


def _coverage(kept, expected_centres, slack=0.35):
    if expected_centres is None or len(expected_centres) == 0:
        return 1.0
    hit = 0
    for (cx, cy) in expected_centres:
        for w in kept:
            mx, my = w.width * slack + 4, w.height * slack + 4
            if w.left - mx <= cx <= w.left + w.width + mx and w.top - my <= cy <= w.top + w.height + my:
                hit += 1
                break
    return hit / float(len(expected_centres))


EDGE_JUNK = "|_~`¦"


def clean_words(words, config):
    """Small fixes for crops: label borders read as '|' or '_' glued to a word
    ('|500'), and lone unsure letters at a line end ('j WET FLOOR') that come
    from the label's edge. Real one-letter words ('a', 'I', '1') read with
    high confidence are kept."""
    lines = {}
    for w in words:
        lines.setdefault((w.block, w.par, w.line), []).append(w)
    out = []
    for key, ws in lines.items():
        ws = sorted(ws, key=lambda w: w.word_num)
        for i, w in enumerate(ws):
            t = w.text.strip(EDGE_JUNK)
            if not t:
                continue
            at_end = i == 0 or i == len(ws) - 1
            if len(t) == 1 and t.isalpha() and at_end and len(ws) > 1 and w.conf < config.sure_word_conf:
                continue
            out.append(w._replace(text=t))
    return out


def combine(word_lists):
    """Word-level voting across recipes: where several readings found a word in
    the same place, keep the most confident version. Different recipes fail on
    different words (one misses the faint first line, another the last), so the
    combination is often better than any single reading."""
    allw = [w for ws in word_lists for w in ws if not is_junk(w.text) and w.conf >= 0]
    allw.sort(key=lambda w: -w.conf)
    chosen = []
    for w in allw:
        if all(_iou(w, c) < 0.3 for c in chosen):
            chosen.append(w)
    chosen.sort(key=lambda w: w.top + w.height / 2.0)
    lines, out = [], []
    for w in chosen:
        cy = w.top + w.height / 2.0
        for ln in lines:
            if abs(ln[0] - cy) < 0.5 * max(w.height, ln[1]):
                ln[2].append(w)
                break
        else:
            lines.append([cy, w.height, [w]])
    for k, (_, _, ws) in enumerate(lines):
        for i, w in enumerate(sorted(ws, key=lambda w: w.left)):
            out.append(Word(w.text, w.conf, 1, 1, k + 1, i + 1, w.left, w.top, w.width, w.height))
    return out


def _iou(a, b):
    ix = max(0, min(a.left + a.width, b.left + b.width) - max(a.left, b.left))
    iy = max(0, min(a.top + a.height, b.top + b.height) - max(a.top, b.top))
    inter = ix * iy
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union else 0.0


def evaluate(words, config, expected_centres=None, cut=False):
    words = clean_words(words, config)
    text = extract_text(words, config)
    kept = confident_words(words, config.min_word_conf) if text else []
    seen = [w for w in words if not is_junk(w.text) and w.conf >= 0]
    total_chars = sum(len(w.text) for w in seen) or 1
    good_chars = sum(len(w.text) * w.conf / 100.0 for w in kept)
    conf = (sum(len(w.text) * w.conf for w in kept) / max(1, sum(len(w.text) for w in kept))) if kept else 0.0
    coverage = _coverage(kept, expected_centres) if kept else 0.0
    score = (good_chars / total_chars) * (0.4 + 0.6 * coverage)
    unsure_numbers = [w.text for w in kept if re.search(r"\d", w.text) and w.conf < config.sure_word_conf]
    if not text:
        status = UNREADABLE
    elif coverage < config.min_coverage or cut:
        status = PARTIAL
    elif is_unclear(words, config):
        status = UNCLEAR
    else:
        status = CLEAR
    return Reading(text, status, score, conf, coverage, cut, unsure_numbers, words)


def better(a, b):
    """The better of two readings (None-safe)."""
    if a is None:
        return b
    if b is None:
        return a
    return b if b.score > a.score else a
