"""
ocr/target_selector.py - decide WHICH text the user means by "read this".

A blind user holds the thing up to the camera; nobody points. So the target is
inferred from how people naturally frame things, with four signals, each 0..1:

  centre   1 at the frame centre, 0 in a corner. People hold what they want
           read in front of the camera.
  size     letter height compared with a comfortable reading size. The held
           object is the closest one, so its letters look biggest; a poster
           across the room has tiny letters.
  sharp    variance of the Laplacian inside the block (see image_preprocessing).
           The camera focuses on what is near; far text is often soft.
  object   1 when the block sits on something YOLO sees (bottle, book, phone,
           laptop...), i.e. a real object rather than a pattern.

score = w_centre*centre + w_size*size + w_sharp*sharp + w_object*object  (config)

This is a hand-tuned "linear scoring" model: simple, explainable and cheap, at
the cost of fixed weights (a learned ranker would need labelled data we don't
have). The weights were checked on the test set in tools/ocr_eval.py.

Decision rule (never guess):
  * blocks on the SAME object are one target (front of one packet).
  * blocks close to each other (gap < cluster_gap_h letter heights) are one
    target too: a poster or a door sign is often split by icons or coloured
    bands into several blocks, and the user means the whole sign.
  * the best block wins only if the runner-up scores below ambiguous_ratio of
    it. Otherwise the choice is AMBIGUOUS and SENSE explains how to frame the
    text it wants (it does not ask "left or right?").
"""

from collections import namedtuple

import numpy as np

from .image_preprocessing import sharpness, to_gray
from .text_regions import _Sets, merge_blocks

Choice = namedtuple("Choice", "block score scores ambiguous")


def block_scores(block, frame_gray, config):
    fh, fw = frame_gray.shape[:2]
    x, y, w, h = block.rect
    cx, cy = x + w / 2.0, y + h / 2.0
    dist = np.hypot((cx - fw / 2.0) / (fw / 2.0), (cy - fh / 2.0) / (fh / 2.0)) / np.sqrt(2)
    centre = float(max(0.0, 1.0 - dist))
    size = float(min(1.0, block.letter_h / 40.0))
    crop = frame_gray[y:y + h, x:x + w]
    sharp = float(min(1.0, sharpness(crop) / 400.0)) if crop.size else 0.0
    obj = 1.0 if block.obj else 0.0
    return {"centre": centre, "size": size, "sharp": sharp, "object": obj}


def score_of(parts, config):
    return (config.w_centre * parts["centre"] + config.w_size * parts["size"]
            + config.w_sharp * parts["sharp"] + config.w_object * parts["object"])


def choose(blocks, frame, config):
    """Choice(block, score, all_scores, ambiguous) or None when there is no text."""
    if not blocks:
        return None
    gray = to_gray(frame)
    # one target per object: merge blocks that sit on the same YOLO object
    by_obj, loose = {}, []
    for b in blocks:
        (by_obj.setdefault(b.obj, []) if b.obj else loose).append(b)
    candidates = loose + [merge_blocks(bs) if len(bs) > 1 else bs[0] for bs in by_obj.values()]
    candidates = cluster_nearby(candidates, config)
    scored = sorted(((score_of(block_scores(b, gray, config), config), b) for b in candidates),
                    key=lambda s: s[0], reverse=True)
    scored = [s for s in scored if s[0] >= config.min_block_score] or scored[:1]
    best_score, best = scored[0]
    ambiguous = len(scored) > 1 and scored[1][0] >= config.ambiguous_ratio * best_score
    return Choice(best, best_score, [s for s, _ in scored], ambiguous)


def _gap(a, b):
    ax, ay, aw, ah = a.rect
    bx, by, bw, bh = b.rect
    dx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    dy = max(0, max(ay, by) - min(ay + ah, by + bh))
    return max(dx, dy)


def cluster_nearby(blocks, config):
    """Merge blocks whose gap is small compared with their letter size."""
    if len(blocks) < 2:
        return blocks
    sets = _Sets(len(blocks))
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            a, b = blocks[i], blocks[j]
            if a.obj and b.obj and a.obj != b.obj:
                continue
            if _gap(a, b) < config.cluster_gap_h * max(a.letter_h, b.letter_h):
                sets.union(i, j)
    out = []
    for g in sets.groups():
        out.append(blocks[g[0]] if len(g) == 1 else merge_blocks([blocks[k] for k in g]))
    return out


def where_in_frame(block, frame_w, frame_h):
    """'in the middle', 'at the top left' ... (camera view) for log/HUD use."""
    x, y, w, h = block.rect
    cx, cy = (x + w / 2.0) / frame_w, (y + h / 2.0) / frame_h
    if 0.3 <= cx <= 0.7 and 0.3 <= cy <= 0.7:
        return "in the middle"
    v = "top " if cy < 0.33 else ("bottom " if cy > 0.67 else "")
    hz = "left" if cx < 0.33 else ("right" if cx > 0.67 else "")
    return ("at the " + (v + hz).strip()) if (v or hz) else "in the middle"
