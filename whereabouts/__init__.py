"""
whereabouts - "Where did I leave it?" for SENSE.

When SENSE sees one of your items with the camera held still, it notes WHERE
the item is in that same picture: the furniture it stands on ("on the table")
and the nearest other object ("to the left of the laptop"). The note is voted
over several sightings and saved in SQLite, so later - even after a restart -
"where is my phone?" can answer:

    "Your phone was on the table, to the left of the laptop, 10 minutes ago."

    config.py      every threshold in one place (WhereaboutsConfig)
    geometry.py    small bounding-box helpers (pure functions)
    furniture.py   "is the item ON the table / desk / bed / couch?"
    landmarks.py   "which object is it next to, and on which side?"
    voting.py      3-of-5 vote, so one noisy frame never changes the answer
    store.py       the item_anchors SQLite table (created if missing, never reset)
    phrasing.py    turns an anchor into the spoken sentence
    tracker.py     WhereaboutsTracker: the only class sense_master.py talks to

Like the ocr/ package, this never opens the camera, never starts a thread and
never calls the network. It only reads what sense_master.py already computed.
"""

from .config import WhereaboutsConfig
from .tracker import WhereaboutsTracker, create

__all__ = ["WhereaboutsConfig", "WhereaboutsTracker", "create"]
