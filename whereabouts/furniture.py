"""
furniture.py - is an item ON a table, desk, bed or couch?

ALGORITHM (one picture, no depth)
An item resting on furniture touches it at the bottom of the item's box. So we
take the item's bottom-centre point and ask: does it fall inside the UPPER part
of the furniture's box (the top surface), not too close to its edges?

    +-----------------+  <- table box top
    |   [phone]       |  phone's bottom-centre lands here: ON the table
    |.................|  <- upper 60% for a table
    |   legs / floor  |  an item whose bottom is here is IN FRONT of the table
    +-----------------+

Also: the item must be clearly smaller than the furniture, and if two DIFFERENT
kinds of furniture both qualify (a phone between a bed and a couch) we say
nothing - an uncertain claim is worse than none for someone who can't check by
looking.

ALTERNATIVES: a depth model (MiDaS) or segmentation masks would be more exact
but need extra models and much more CPU per frame. A 2D box test is an
approximation, which is why the result is voted over several sightings
(voting.py) and why ambiguous geometry returns None.
"""

from . import geometry as g


def surface_under(item_box, furniture, frame_h, cfg):
    """item_box: (x1,y1,x2,y2). furniture: list of (category, box).
    Returns the surface category ("table") or None when unsure."""
    cx, bottom = g.bottom_centre(item_box)
    hits = []
    for category, fbox in furniture:
        frac = cfg.surfaces.get(category)
        if frac is None:
            continue
        fw, fh = g.width(fbox), g.height(fbox)
        if fw <= 0 or fh <= 0:
            continue
        if g.area(item_box) > cfg.max_item_to_surface_area * g.area(fbox):
            continue
        if g.width(item_box) > cfg.max_item_to_surface_width * fw:
            continue
        if fbox[3] >= frame_h - 2:      # cut off by the picture's bottom edge
            frac = max(frac, cfg.cut_off_surface_fraction)
        side = cfg.surface_side_margin * fw
        if not (fbox[0] + side <= cx <= fbox[2] - side):
            continue
        top = fbox[1] + cfg.surface_top_margin * fh
        if not (top <= bottom <= fbox[1] + frac * fh):
            continue
        hits.append(category)
    kinds = set(hits)
    if len(kinds) != 1:          # nothing, or two different kinds: ambiguous
        return None
    return kinds.pop()
