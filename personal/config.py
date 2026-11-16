"""All personal-item identity settings in one place, with the reason for each value.

The colour signature is the 16x20 hue/saturation histogram that sense_master
already computes (get_unified_chroma_signature). "Similarity" below is the
correlation of two such histograms (1.0 = identical, 0 = unrelated).
"""

from dataclasses import dataclass, field


@dataclass
class PersonalConfig:
    # --- appearance --------------------------------------------------------
    # The old matcher accepted 0.55, which two white cups pass easily. 0.75
    # still accepts the same cup under normal light changes (measured on the
    # cups, bottles and phones in tests/test_personal_items.py).
    match_min: float = 0.75
    # After a restart there is no position to check, so the look has to be
    # even closer before we say "your cup" again.
    restart_match_min: float = 0.80
    # After a restart the heading is not comparable with the last run. By
    # default an item is then never confirmed automatically ("looks like
    # yours" until "this is my cup" again). True = confirm colourful items seen
    # at the old place - nicer, but an identical cup that was never seen next
    # to yours could be taken for it if SENSE restarts facing another way.
    confirm_after_restart: bool = False
    # An item counts as colourless (white/grey/black) when this share of its
    # colour histogram is near "no colour"; see matcher.is_neutral.
    neutral_mass: float = 0.60
    # "Looks like yours, but I can't be sure" starts here.
    lookalike_min: float = 0.65
    # The candidate must look clearly more like YOUR item than like any other
    # object of that kind that SENSE has seen next to it (a look-alike).
    margin: float = 0.10
    # Two personal items of the same kind: the winner must lead by this much,
    # otherwise neither is assigned (we never guess between them).
    item_margin: float = 0.08

    # --- position (world angle, same run only) ------------------------------
    # The heading estimate can be ~20 degrees off after a quick turn
    # (heading_fusion.py), so "near where it was" allows 25 degrees.
    near_deg: float = 25.0
    # A locked track whose object jumped this far while the camera was turning
    # is treated as a possible ByteTrack id switch: the label is dropped.
    jump_deg: float = 25.0
    # ... or when another track of the same kind is this much closer to where
    # the item was before the turn
    resume_slack_deg: float = 5.0

    # --- tracks ------------------------------------------------------------
    # ByteTrack ids are not identities: a track unseen for longer than this
    # loses its personal label and must be re-confirmed.
    lock_gap_s: float = 1.5
    # An item counts as "held" by a track seen this recently, so no second
    # track can take its name at the same time.
    held_s: float = 1.0
    # A NEW track needs `confirm_needed` of its last `confirm_window` checks
    # to agree before it is called "your cup" (one lucky frame is not enough).
    confirm_window: int = 8
    confirm_needed: int = 5
    # A locked track that keeps looking unlike the item (camera still) for
    # this many checks in a row loses its label.
    doubt_frames: int = 6
    # A track's look is the average of its last few sharp frames; decide
    # nothing until there are this many (one frame's colours are noisy).
    min_samples: int = 3
    doubt_below: float = 0.45

    # --- look-alikes ---------------------------------------------------------
    max_lookalikes: int = 6
    lookalike_merge: float = 0.85          # a near-duplicate is averaged into the old entry
    lookalike_every_s: float = 0.3         # at most ~3 look-alike samples per second

    # --- learning ------------------------------------------------------------
    adapt_rate: float = 0.05               # slow update of the item's look while locked
    save_every_s: float = 2.0              # SQLite writes per item at most this often

    # --- teaching ------------------------------------------------------------
    # "this is my cup" with two cups of nearly the same size in view: ask the
    # user to hold theirs closer instead of guessing.
    teach_runner_up_ratio: float = 0.80

    # --- answering -------------------------------------------------------------
    visible_s: float = 1.5                 # "can see it now" window for answers
    ambiguous_s: float = 3.0               # "two cups look like yours" lasts this long

    # Spoken words -> SENSE category (sense_master's ALLOWED_ITEMS values)
    synonyms: dict = field(default_factory=lambda: {
        "mug": "cup", "glass": "cup", "cups": "cup",
        "mobile": "phone", "cellphone": "phone", "iphone": "phone", "phones": "phone",
        "backpack": "bag", "handbag": "bag", "purse": "bag", "bags": "bag",
        "bottles": "bottle", "flask": "bottle",
        "computer": "laptop", "macbook": "laptop", "laptops": "laptop",
        "chairs": "chair", "seat": "chair", "books": "book", "tv remote": "remote",
    })
