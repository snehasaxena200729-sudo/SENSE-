"""
landmarks.py - which object is the item next to, and on which side?

ALGORITHM (one picture)
For every other object in the SAME frame:
  1. skip people/animals (they move) and the item itself;
  2. skip it if the two boxes overlap a lot - then the item may be in front
     of it, behind it or on top of it, and a picture can't tell which;
  3. skip it if their bottoms are at very different heights - they are not
     standing on the same level (one is on a shelf, or much further away);
  4. skip it if the sideways gap is too big to call them "near";
  5. cost = sideways gap, multiplied by a penalty for things that often move
     (a cup is a worse landmark than a laptop). The cheapest one wins.
Then the side: boxes overlapping sideways -> "next to", otherwise "left of" /
"right of" by comparing centres. The camera is worn or held by the user, so
the camera's left is the user's left.

Because both boxes come from the same picture, this does not depend on how far
the user has turned (spatial.py's heading drifts and resets every run), so the
relation stays true after a restart. That is the main difference from
sense_master.memory_relation(), which compares positions saved at different
times and only works within one run.
"""

from . import geometry as g


def nearest_landmark(target_name, target_box, others, frame_w, cfg):
    """others: list of (name, category, box). Returns (name, category, side)
    with side in {"left of", "right of", "next to"}, or None."""
    best, best_cost = None, None
    tw, th = g.width(target_box), g.height(target_box)
    if tw <= 0 or th <= 0:
        return None
    for name, category, box in others:
        if name.lower() == target_name.lower() or category in cfg.never_landmarks:
            continue
        lw, lh = g.width(box), g.height(box)
        if lw <= 0 or lh <= 0:
            continue
        inter = g.intersection(target_box, box)
        if inter > cfg.max_overlap_of_target * g.area(target_box) or \
                inter > cfg.max_overlap_of_target * g.area(box):
            continue                       # in front / behind / on top: can't tell
        if abs(target_box[3] - box[3]) > cfg.max_level_ratio * max(th, lh):
            continue                       # not on the same level
        gap = g.horizontal_gap(target_box, box)
        if gap > min(cfg.max_gap_frame_fraction * frame_w, cfg.max_gap_box_ratio * max(tw, lw)):
            continue
        cost = (gap + abs(target_box[3] - box[3]) * 0.5) / float(frame_w)
        if category not in cfg.stable_landmarks:
            cost *= cfg.movable_landmark_penalty
        if best_cost is None or cost < best_cost:
            best, best_cost = (name, category, box), cost
    if best is None:
        return None
    name, category, box = best
    overlap = g.horizontal_overlap(target_box, box) / min(tw, g.width(box))
    if overlap >= cfg.next_to_overlap:
        side = "next to"
    else:
        side = "left of" if g.centre_x(target_box) < g.centre_x(box) else "right of"
    return name, category, side
