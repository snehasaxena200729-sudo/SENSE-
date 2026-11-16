"""
interactive/snapshot.py - a read-only picture of the last YOLO frame.

The main loop calls see() once per confirmed detection and frame_done() once
per YOLO frame. Only REFERENCES are stored (the crop is a view into that
frame's image, no copy), so this costs microseconds per frame. Path B never
writes back into anything SENSE owns.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional

import spatial

from .colour import is_cut_off


@dataclass
class Seen:
    tid: int
    cls: str                 # "bottle", "person", "chair" ...
    label: str               # SENSE's display name: "White Bottle", "My Bottle", "Person", "Sneha"
    person: Optional[str]    # face-verified name, else None (never guessed)
    box: tuple
    crop: Any
    dist_phrase: str         # distance.phrase(): a range, a band or "distance unclear"
    dist_m: Optional[float]
    angle: float             # degrees, + = right
    cut_off: bool

    @property
    def direction(self):
        return spatial.direction_phrase(self.angle)

    @property
    def side(self):
        d = self.direction
        return {"in front of you": "ahead", "on your left": "left",
                "on your right": "right"}.get(d, "behind")


@dataclass
class Snapshot:
    time: float = 0.0
    frame_w: int = 1280
    frame_h: int = 720
    items: List[Seen] = field(default_factory=list)
    gray: Any = None         # that frame's grey image (a reference, read only on a question)
    _scene_v: Optional[float] = None

    def scene_brightness(self):
        """Median brightness of the whole frame (0-255), computed only when asked."""
        if self._scene_v is None and self.gray is not None:
            import numpy as np
            self._scene_v = float(np.median(self.gray[::8, ::8]))
        return self._scene_v

    def age(self, now):
        return now - self.time if self.time else 1e9


class SnapshotBuilder:
    def __init__(self):
        self.current = Snapshot()
        self._building = []

    def see(self, tid, cls, label, person, box, crop, dist_phrase, dist_m, frame_w, frame_h):
        x1, y1, x2, y2 = box
        self._building.append(Seen(int(tid), cls, label, person, tuple(box), crop, dist_phrase, dist_m,
                                   spatial.pixel_to_angle((x1 + x2) / 2.0, frame_w),
                                   is_cut_off(box, frame_w, frame_h)))

    def frame_done(self, now, frame_w, frame_h, gray=None):
        self.current = Snapshot(now, frame_w, frame_h, self._building, gray)
        self._building = []
