"""
interactive/colour.py - "what colour is it?", with an honest confidence.

The colour NAME comes from SENSE's existing get_dominant_color_name()
(passed in as `name_fn`, so there is exactly one colour classifier). This file
only adds what that function cannot tell you: HOW SURE the answer is.

  1. Look at the middle 60% of the detection box. A YOLO box also contains
     table, wall and hand; the object itself is mostly in the middle.
  2. Refuse when the evidence is bad: box too small, box cut off by the frame
     edge (part of the object is missing), or too dark.
  3. Measure agreement: among the strongly coloured pixels, what share falls
     in the winning hue band? Under 55% with a strong runner-up -> "blue or
     green". Near the neutral/colour threshold -> "white or yellow".
  4. Cross-check the colour SENSE voted over many frames when it first named
     the object ("White Bottle"). Disagreement is reported, not hidden.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config as C
from .text import COLOUR_WORDS

# Same hue bands (OpenCV hue 0-179) as get_dominant_color_name() in sense_master.py
_BANDS = [("red", 0, 10), ("orange", 10, 25), ("yellow", 25, 35), ("green", 35, 85),
          ("cyan", 85, 100), ("blue", 100, 130), ("purple", 130, 170), ("red", 170, 180)]


@dataclass
class ColourReading:
    status: str                    # "clear" | "ambiguous" | "partial" | "too_small" | "too_dark" | "none"
    name: Optional[str] = None     # the existing classifier's answer on the core of the box
    alternative: Optional[str] = None
    share: float = 0.0


def is_cut_off(box, frame_w, frame_h, edge=C.EDGE_PX):
    x1, y1, x2, y2 = box
    return x1 <= edge or y1 <= edge or x2 >= frame_w - edge or y2 >= frame_h - edge


def _core(crop, frac=C.CORE_FRACTION):
    h, w = crop.shape[:2]
    dy, dx = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    return crop[dy:h - dy, dx:w - dx]


def _hsv(crop):
    """BGR -> HSV without needing OpenCV here (OpenCV is used when present)."""
    try:
        import cv2
        return cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    except Exception:          # pragma: no cover - only on machines without cv2
        rgb = crop[..., ::-1].astype(np.float32) / 255.0
        mx, mn = rgb.max(-1), rgb.min(-1)
        d = mx - mn + 1e-9
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4))
        h = np.where(mx - mn < 1e-6, 0, h) * 30
        s = np.where(mx > 0, (mx - mn) / (mx + 1e-9), 0) * 255
        return np.dstack([h, s, mx * 255]).astype(np.uint8)


def hue_shares(hsv):
    """{colour: share} among strongly coloured pixels (the existing thresholds)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (s >= 70) & (v >= 60)
    n = int(np.count_nonzero(mask))
    if n == 0:
        return {}, 0.0
    hues = h[mask]
    shares = {}
    for name, lo, hi in _BANDS:
        c = int(np.count_nonzero((hues >= lo) & (hues < hi)))
        if c:
            shares[name] = shares.get(name, 0.0) + c / n
    return shares, n / float(mask.size)


def read_colour(crop, name_fn, box=None, frame_w=None, frame_h=None, scene_v=None):
    if crop is None or getattr(crop, "size", 0) == 0:
        return ColourReading("none")
    h, w = crop.shape[:2]
    if h < C.MIN_CROP_PX or w < C.MIN_CROP_PX:
        return ColourReading("too_small")
    core = _core(crop)
    hsv = _hsv(core)
    # A black object is dark too, so "too dark" needs a dark ROOM (whole frame),
    # not just a dark crop. Without the frame, only near-total darkness counts.
    crop_v = float(np.median(hsv[..., 2]))
    if crop_v < C.DARK_V and (scene_v is None and crop_v < 12 or scene_v is not None and scene_v < C.DARK_SCENE_V):
        return ColourReading("too_dark")
    name = name_fn(core) or None
    shares, coloured = hue_shares(hsv)
    alternative, share, status = None, 1.0, "clear"
    if name in ("white", "black", "gray"):
        # near the existing 15% "is it coloured at all?" threshold -> say both
        if coloured >= 0.10 and shares:
            alternative = max(shares, key=shares.get)
            status = "ambiguous"
        share = 1.0 - coloured
    elif name:
        share = shares.get(name, 0.0)
        ranked = sorted(((v, k) for k, v in shares.items()), reverse=True)
        if share < C.AMBIGUOUS_SHARE and len(ranked) > 1 and ranked[1][0] >= C.RUNNER_UP_SHARE:
            # two real hues (half blue, half green): the median hue of the existing
            # classifier lands BETWEEN them ("cyan"), so name the two that are there
            name, alternative, status = ranked[0][1], ranked[1][1], "ambiguous"
            share = ranked[0][0]
    else:
        return ColourReading("none")
    if box is not None and frame_w and is_cut_off(box, frame_w, frame_h):
        status = "partial" if status == "clear" else "ambiguous"
    return ColourReading(status, name, alternative, round(share, 2))


def voted_colour(label):
    """'White Bottle' -> 'white' (the colour SENSE voted when it named the object)."""
    for w in (label or "").lower().split():
        if w in COLOUR_WORDS:
            return w
    return None
