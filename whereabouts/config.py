"""All whereabouts settings in one place, with the reason for each value."""

from dataclasses import dataclass, field


@dataclass
class WhereaboutsConfig:
    # --- what can be anchored -------------------------------------------
    # Items people misplace. These are SENSE category names (sense_master's
    # ALLOWED_ITEMS values); items you taught ("my bottle") carry their base
    # category, so they are included automatically.
    item_categories: frozenset = frozenset(
        {"phone", "remote", "cup", "bottle", "book", "laptop", "mouse", "bag"})

    # Surfaces and the part of their box (from the top) where a resting item's
    # bottom edge can be. From a standing or sitting camera the top of a table
    # is the upper part of its box; the lower part is legs and floor, where an
    # item would be IN FRONT of the table, not on it. A couch box includes the
    # backrest, so its seat is lower down. "desk" has no class in the current
    # COCO model (desks are usually detected as "dining table" -> "table"), but
    # it is supported for any detector that reports it.
    surfaces: dict = field(default_factory=lambda: {
        "table": 0.60, "desk": 0.60, "bed": 0.75, "couch": 0.80})
    # When the furniture box is cut off by the bottom of the picture, we only
    # see its top, so almost all of the visible box is surface.
    cut_off_surface_fraction: float = 0.95
    surface_top_margin: float = 0.04       # bottom must be clearly INSIDE the box (not behind it)
    surface_side_margin: float = 0.03      # ... and not on its very edge
    max_item_to_surface_area: float = 0.35 # a "phone" as big as the bed is not on it
    max_item_to_surface_width: float = 0.60

    # --- landmarks -------------------------------------------------------
    # People and animals move, so "left of Sneha" is useless an hour later.
    never_landmarks: frozenset = frozenset({"person", "dog", "cat"})
    # Things that rarely move make better landmarks than other small items.
    stable_landmarks: frozenset = frozenset(
        {"laptop", "tv", "plant", "chair", "couch", "bed", "table", "desk", "suitcase",
         "bench", "bicycle"})
    movable_landmark_penalty: float = 1.4  # cost multiplier for cups, bottles, phones...
    max_gap_frame_fraction: float = 0.20   # side gap at most 20% of the picture width
    max_gap_box_ratio: float = 1.5         # ... and at most 1.5x the wider box
    max_level_ratio: float = 0.5           # bottoms at a similar height = same surface level
    max_overlap_of_target: float = 0.5     # boxes this overlapped: in front / on top? -> don't say
    next_to_overlap: float = 0.25          # side-by-side overlap that reads as "next to"

    # --- voting ----------------------------------------------------------
    window: int = 5                        # last 5 samples ...
    needed: int = 3                        # ... 3 must agree
    sample_interval_s: float = 0.4         # samples ~0.4 s apart, so 5 span ~2 s
    reset_gap_s: float = 10.0              # not seen for 10 s: start a fresh vote
    moved_deg: float = 12.0                # same move threshold as events.py

    # --- storage and answers ---------------------------------------------
    refresh_s: float = 5.0                 # rewrite an unchanged anchor at most every 5 s
    stale_gap_s: float = 30.0              # anchor older than the last sighting by this: "earlier"
