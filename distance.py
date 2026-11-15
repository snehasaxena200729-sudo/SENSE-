"""
distance.py - How far away is it? Honest monocular distance (audit fix 2).

THE GEOMETRY (pinhole camera)
-----------------------------
An object of real size S metres at distance D appears s pixels big:

        s = f * S / D        so        D = f * S / s

f is the focal length IN PIXELS. It depends on the lens AND the resolution:

        f = (frame_width / 2) / tan(HFOV / 2)

Old SENSE used f = 600 px for every resolution. With the 65 degree view
assumed elsewhere in the code, a 1280 px wide frame has f ~ 1005 px, so
distances came out about 40% short. Worse: when you stand close, your box is
CUT OFF by the edge of the frame. Its height is then the frame height, not
your height, so the estimate stopped changing (about 5 feet) however close
you were.

WHY THIS CAN ONLY EVER BE APPROXIMATE
-------------------------------------
One camera cannot measure depth. It can only infer it from an ASSUMED real
size S, so the error in D is at least the error in S:
  * eye-to-eye distance (adults ~6.3 cm, varies ~+-10%)  -> best cue for people
  * person height (posture! sitting vs standing)          -> rough
  * shoulder/box width of a person                         -> rough
  * typical object sizes (a "cup" can be 7 or 12 cm)       -> rough
plus the focal length (unknown until calibrated, +-15% from the FOV guess).

So every estimate here carries a low..high range, and SENSE speaks a RANGE
or a coarse band, never a precise-sounding number. If no trustworthy cue
exists it says the distance is unclear.
"""

import math
from dataclasses import dataclass

DEFAULT_HFOV_DEG = 65.0      # same assumption as spatial.CAMERA_FOV_DEG until calibrated
FOCAL_UNCERTAINTY = 0.15     # +-15% from the guessed field of view
CALIBRATED_FOCAL_UNCERTAINTY = 0.05
EDGE_MARGIN_FRAC = 0.02      # a box within 2% of the frame edge is treated as cut off
                             # (YOLO boxes of cut-off people often stop a few px short)

M_TO_FT = 3.28084
ARM_REACH_M = 0.8
FAR_M = 4.0

# Reference sizes: (metres, relative uncertainty). For objects this is the
# LONGEST side, compared with the longer side of the box.
EYE_DISTANCE = (0.063, 0.10)        # interpupillary distance, adults
PERSON_STANDING = (1.70, 0.12)
PERSON_UNKNOWN_POSTURE = (1.40, 0.30)   # sitting or standing, can't tell
PERSON_WIDTH = (0.50, 0.35)         # shoulders + arms, facing the camera (rough: arms, pose)
OBJECT_SIZES = {
    "phone": (0.15, 0.20), "remote": (0.18, 0.25), "bottle": (0.24, 0.30),
    "cup": (0.10, 0.30), "laptop": (0.33, 0.15), "book": (0.23, 0.30),
    "mouse": (0.11, 0.20), "chair": (0.90, 0.20), "couch": (1.90, 0.25),
    "bed": (2.00, 0.15), "table": (1.20, 0.35), "plant": (0.50, 0.50),
    "tv": (1.00, 0.35), "dog": (0.70, 0.50), "cat": (0.45, 0.30),
    "suitcase": (0.65, 0.25), "bench": (1.50, 0.30), "bicycle": (1.70, 0.15),
}


@dataclass
class Estimate:
    metres: float = None      # best guess, None if unknown
    low: float = None         # plausible range
    high: float = None
    cue: str = "none"         # which measurement was used
    upper_bound_only: bool = False   # box cut off: the object is CLOSER than `high`

    @property
    def known(self):
        return self.high is not None


def focal_px(frame_width, hfov_deg=DEFAULT_HFOV_DEG, focal_ratio=None):
    """Focal length in pixels for this resolution.

    focal_ratio = f / frame_width, measured once by tools/validate_distance.py
    (resolution independent, so it survives a change of capture size)."""
    if focal_ratio:
        return focal_ratio * frame_width
    return (frame_width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def from_size(real, size_px, f_px, focal_unc=FOCAL_UNCERTAINTY, cue="size"):
    """Pinhole estimate with an error range. real = (metres, relative uncertainty)."""
    if size_px <= 1:
        return Estimate()
    size_m, size_unc = real
    d = f_px * size_m / size_px
    unc = math.hypot(size_unc, focal_unc)          # independent errors add in quadrature
    return Estimate(d, d * (1 - unc), d * (1 + unc), cue)


def clipped_edges(box, frame_w, frame_h, margin_frac=EDGE_MARGIN_FRAC):
    x1, y1, x2, y2 = box
    mx, my = max(4, margin_frac * frame_w), max(4, margin_frac * frame_h)
    edges = set()
    if x1 <= mx:
        edges.add("left")
    if y1 <= my:
        edges.add("top")
    if x2 >= frame_w - mx:
        edges.add("right")
    if y2 >= frame_h - my:
        edges.add("bottom")
    return edges


def person_distance(box, frame_w, frame_h, f_px, eye_px=None, focal_unc=FOCAL_UNCERTAINTY):
    """Best available distance cue for a person box.

    1. eye distance from a face detector (when a frontal face is visible)
    2. full body height, if the box is not cut off at the top or bottom
    3. body width, if cut off vertically but not at the sides
    4. otherwise only an upper bound (the person is closer than this)
    """
    if eye_px and eye_px > 8:
        return from_size(EYE_DISTANCE, eye_px, f_px, focal_unc, "eyes")
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    edges = clipped_edges(box, frame_w, frame_h)
    if not edges & {"top", "bottom"}:
        size = PERSON_STANDING if h > 2.2 * w else PERSON_UNKNOWN_POSTURE
        return from_size(size, h, f_px, focal_unc, "height")
    if not edges & {"left", "right"} and w > 0.08 * frame_w:
        return from_size(PERSON_WIDTH, w, f_px, focal_unc, "width")
    # Cut off on all useful sides: the visible part is SMALLER than the real
    # person, so the person is CLOSER than a tall person would be at this size.
    bound = f_px * 1.9 / max(h, 1)
    return Estimate(None, None, bound, "upper-bound", upper_bound_only=True)


def object_distance(category, box, frame_w, frame_h, f_px, focal_unc=FOCAL_UNCERTAINTY):
    real = OBJECT_SIZES.get(category)
    if real is None:
        return Estimate()
    x1, y1, x2, y2 = box
    longest = max(x2 - x1, y2 - y1)
    est = from_size(real, longest, f_px, focal_unc, "object size")
    if clipped_edges(box, frame_w, frame_h):
        # part of the object is outside the view: it is closer than this
        return Estimate(None, None, est.high, "upper-bound", upper_bound_only=True)
    # A tilted object looks shorter than its longest side -> it may be closer
    # than the estimate, never further. Widen the range downwards.
    est.low = est.low * 0.7
    return est


def _feet(m):
    return max(1, int(round(m * M_TO_FT)))


def phrase(est):
    """Words for speech. Coarse on purpose: a range, a band, or 'unclear'."""
    if not est.known:
        return "distance unclear"
    if est.high <= ARM_REACH_M:
        return "within arm's reach"
    if est.upper_bound_only:
        return f"closer than about {_feet(est.high)} feet"
    if est.low >= FAR_M:
        return "far away"
    lo, hi = _feet(est.low), _feet(est.high)
    if lo == hi:
        return f"about {lo} feet away"
    return f"about {lo} to {hi} feet away"


def feet_or_none(est):
    """Single number for the database (None when unknown)."""
    return est.metres * M_TO_FT if est.metres is not None else None


def is_close(est, metres=1.5):
    """True if the object could plausibly be nearer than `metres`."""
    return est.known and (est.low if est.low is not None else 0.0) < metres
