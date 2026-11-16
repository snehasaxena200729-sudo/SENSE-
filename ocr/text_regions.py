"""
ocr/text_regions.py - join word boxes into text BLOCKS (one label, one sign).

EAST finds single words. A medicine label is many words on a few lines, and
SENSE should read the label as one thing, not pick one word of it. Grouping:

  1. LINES   two words are on the same line when their centres are at about
             the same height (along the text slope) and the gap between them
             is small compared with the letter height (line_gap_h).
  2. BLOCKS  two lines belong to one block when they overlap horizontally and
             the vertical gap is small (block_gap_h letter heights), and their
             letter sizes are similar (max_height_ratio). A headline in big
             letters and small print under it still join if they touch.

Both steps use union-find ("disjoint sets"): every word starts as its own
group, and each "these two belong together" merges two groups. It handles
chains (A next to B next to C) without caring about the order of the boxes.

Letter height (the short side of a word box) is the natural unit: "close" for
2 cm letters on a sign is very different from "close" for 3 mm letters on a
blister strip, and measuring gaps in letter heights treats both the same way.
"""

from collections import namedtuple

import cv2
import numpy as np

# boxes: the TextBox words in this block; rect: (x, y, w, h); quad: 4 corners of
# the tightest rotated rectangle around all words; angle: median text slope;
# letter_h: median letter height in px; lines: number of text lines;
# obj: name of the YOLO object the block sits on, or None.
Block = namedtuple("Block", "boxes rect quad angle letter_h lines obj")


class _Sets:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)

    def groups(self):
        out = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())


def _centre(box):
    return box.quad.mean(axis=0)


def _same_line(a, b, cfg):
    h = min(a.height, b.height)
    if max(a.height, b.height) > cfg.max_height_ratio * h:
        return False
    ang = np.deg2rad((a.angle + b.angle) / 2.0)
    d = (_centre(b) - _centre(a))
    along = abs(d[0] * np.cos(ang) + d[1] * np.sin(ang))           # distance along the line
    across = abs(-d[0] * np.sin(ang) + d[1] * np.cos(ang))         # distance across it
    half_lengths = (a.rect[2] + b.rect[2]) / 2.0
    return across < 0.6 * h and along - half_lengths < cfg.line_gap_h * h


def _rect_of(boxes):
    pts = np.vstack([b.quad for b in boxes])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return pts, (int(max(0, x0)), int(max(0, y0)), int(x1 - max(0, x0)), int(y1 - max(0, y0)))


def _lines_touch(la, lb, cfg):
    _, (ax, ay, aw, ah) = _rect_of(la)
    _, (bx, by, bw, bh) = _rect_of(lb)
    ha = float(np.median([b.height for b in la]))
    hb = float(np.median([b.height for b in lb]))
    h = min(ha, hb)
    if max(ha, hb) > cfg.max_height_ratio * h * 1.5:
        return False
    overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
    if overlap_x < 0.2 * min(aw, bw):
        return False
    gap_y = max(ay, by) - min(ay + ah, by + bh)
    return gap_y < cfg.block_gap_h * h


def _object_for(rect, objects):
    """The smallest YOLO box that contains the block's centre."""
    x, y, w, h = rect
    cx, cy = x + w / 2.0, y + h / 2.0
    best = None
    for name, (x1, y1, x2, y2) in objects or []:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = (x2 - x1) * (y2 - y1)
            if best is None or area < best[0]:
                best = (area, name)
    return best[1] if best else None


def group_blocks(boxes, config, objects=None):
    """TextBox words -> list of Block, biggest first."""
    if not boxes:
        return []
    n = len(boxes)
    words = _Sets(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _same_line(boxes[i], boxes[j], config):
                words.union(i, j)
    lines = [[boxes[k] for k in g] for g in words.groups()]
    line_sets = _Sets(len(lines))
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if _lines_touch(lines[i], lines[j], config):
                line_sets.union(i, j)
    blocks = []
    for g in line_sets.groups():
        members = [b for k in g for b in lines[k]]
        pts, rect = _rect_of(members)
        quad = cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32))).astype(np.float32)
        blocks.append(Block(members, rect, quad, float(np.median([b.angle for b in members])),
                            float(np.median([b.height for b in members])), len(g),
                            _object_for(rect, objects)))
    blocks.sort(key=lambda b: b.rect[2] * b.rect[3], reverse=True)
    return blocks


def merge_blocks(blocks):
    """Several blocks on the same object are one target (front of one packet)."""
    members = [b for blk in blocks for b in blk.boxes]
    pts, rect = _rect_of(members)
    quad = cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32))).astype(np.float32)
    return Block(members, rect, quad, float(np.median([b.angle for b in members])),
                 float(np.median([b.height for b in members])), sum(b.lines for b in blocks),
                 next((b.obj for b in blocks if b.obj), None))
