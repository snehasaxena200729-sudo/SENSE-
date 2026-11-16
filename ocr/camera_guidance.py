"""
ocr/camera_guidance.py - when reading fails, say WHY in one short sentence.

"I can't read it" does not help a blind user; "move it closer" does. Each
problem is measured from pixels, then the single most useful tip is spoken
(one tip at a time: a list of five instructions is hard to follow by ear).

Problem            measurement                                  tip
-----------------  -------------------------------------------  ----------------------------
no text at all     detector found nothing; YOLO sees an object  turn the label to the camera
too dark           mean brightness of the block/frame           more light
glare              share of blown-out (>=250) pixels in block   tilt away from the light
blurry             Laplacian variance of the block              hold it still
letters too small  detector letter height in px                 move closer
cut off            block touches the frame edge                 move it to the middle / back
tilted             text slope from the detector, or trapezoid   turn it to face the camera
letters too big    letter height in px                          move it further away
several candidates two blocks scored almost the same            hold the one you want in the middle

Order matters: light and blur ruin everything else, so they come first; size
comes before tilt because tiny text also makes the angle unreliable.

(Named camera_guidance.py so it does not clash with guidance.py, find mode.)
"""

import numpy as np

from .image_preprocessing import sharpness, to_gray

HELD_OBJECTS = {"bottle", "book", "phone", "cup", "remote", "laptop", "mouse"}

TIP_NO_TEXT = ("I can't find any text. Hold the label about 30 centimetres from the camera, "
               "facing it, in good light.")
TIP_LABEL_AWAY = "I can see a {obj}, but no text on it. Turn it slowly so the label faces the camera."
TIP_DARK = "It's too dark to read. Turn on a light or move nearer a window."
TIP_GLARE = "There is a shine on the label. Tilt it a little away from the light."
TIP_BLUR = "The text is blurry. Hold it still for a second."
TIP_SMALL = "The text is too small. Move it closer to the camera."
TIP_CUT = "Part of the text is outside the picture. Move it towards the middle, or a little further back."
TIP_TILT = "The label is at an angle. Turn it so it faces the camera straight on."
TIP_BIG = "It's too close. Move it a little further away."
TIP_AMBIGUOUS = ("I can see text in more than one place. Hold the one you want in the middle "
                 "of the camera view, a little closer.")
TIP_GENERIC = "Hold it steady about 30 centimetres from the camera, facing it, in good light."


def touches_edge(rect, frame_w, frame_h, margin):
    x, y, w, h = rect
    mx, my = frame_w * margin, frame_h * margin
    return x <= mx or y <= my or x + w >= frame_w - mx or y + h >= frame_h - my


def measure(frame, block, config):
    gray = to_gray(frame)
    fh, fw = gray.shape[:2]
    if block is None:
        region = gray
    else:
        x, y, w, h = block.rect
        region = gray[y:y + h, x:x + w]
        if region.size == 0:
            region = gray
    m = {"mean": float(np.mean(region)),
         "glare": float(np.mean(region >= 250)),
         "sharp": sharpness(region)}
    if block is not None:
        m.update(letter_h=block.letter_h, angle=abs(block.angle),
                 cut=touches_edge(block.rect, fw, fh, config.edge_margin))
    return m


def no_text_tip(frame, objects, config):
    """Nothing that looks like text was found."""
    m = measure(frame, None, config)
    if m["mean"] < config.dark_mean:
        return TIP_DARK
    fh, fw = frame.shape[:2]
    for name, (x1, y1, x2, y2) in objects or []:
        cx, cy = (x1 + x2) / 2.0 / fw, (y1 + y2) / 2.0 / fh
        if name in HELD_OBJECTS and 0.2 <= cx <= 0.8 and 0.15 <= cy <= 0.85:
            return TIP_LABEL_AWAY.format(obj=name)
    if m["sharp"] < config.blur_threshold:
        return TIP_BLUR
    return TIP_NO_TEXT


def failed_read_tip(frame, block, config, perspective=False):
    """Text was found but could not be read (fully)."""
    m = measure(frame, block, config)
    if m["mean"] < config.dark_mean:
        return TIP_DARK
    if m["glare"] > config.glare_fraction:
        return TIP_GLARE
    if m["sharp"] < config.blur_threshold:
        return TIP_BLUR
    if m["letter_h"] < config.min_letter_px:
        return TIP_SMALL
    if m["cut"]:
        return TIP_CUT
    if m["angle"] > config.tilt_deg or perspective:
        return TIP_TILT
    if m["letter_h"] > config.max_letter_px:
        return TIP_BIG
    return TIP_GENERIC
