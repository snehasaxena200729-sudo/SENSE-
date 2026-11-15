"""
camera_source.py - which camera SENSE reads from, and keeping it alive.

Why this exists
---------------
SENSE used to call cv2.VideoCapture(0) and stop the whole app the first time
a frame could not be read. On a Mac with Camo Studio installed there are at
least two cameras ("FaceTime HD Camera" and "Camo"), and index 0 is not
"the built-in one":

    OpenCV's macOS backend (modules/videoio/src/cap_avfoundation_mac.mm)
    lists every video device and SORTS THEM BY uniqueID (a string), then
    picks devices[index].

So the index of a camera depends on the uniqueID strings of every camera that
is plugged in at that moment. Adding Camo (or an iPhone, or OBS) can move the
built-in camera from 0 to 1. Hard-coding an index is therefore a guess.

How a camera is chosen (resolve())
----------------------------------
1. List the cameras by NAME with their uniqueIDs, the same list OpenCV uses:
   pyobjc's AVFoundation if installed (exact), otherwise macOS's own
   `system_profiler SPCameraDataType -json` (built in, no install).
2. Sort the uniqueIDs exactly like OpenCV does -> the index of each name.
3. If tools/find_cameras.py saved a verified index for this camera AND the
   name at that index still matches, that index wins (a human looked at it).
4. Open it and check that real frames arrive.
If the camera you asked for is not there (Camo Studio closed, phone unplugged)
you get a clear message and SENSE falls back to the built-in camera instead of
going blind.

At run time (CameraSource.read())
---------------------------------
- Every frame comes out as 3-channel BGR uint8 (what YOLOv8, optical flow and
  the face models expect), and no wider than max_width (Camo can send 1080p or
  4K; YOLO resizes to 640 anyway, so big frames only cost time).
- A failed read does NOT end SENSE any more: it reports "camera lost", tries to
  reopen every RETRY_S seconds (no extra thread: one open attempt per retry,
  from the main loop) and reports when it is back.
- A picture that does not change AT ALL for FREEZE_S seconds is reported: that
  is what Camo shows when Studio is running but the phone stopped sending
  video. A live camera always has some sensor/compression noise.

Trade-off: probing every index (open each camera and look) is the only 100%
certain mapping, but it turns on each camera for about a second and is slow, so
it is done once, interactively, by tools/find_cameras.py, not at every start.
"""
import json
import subprocess
import sys
import time

import cv2
import numpy as np

BUILTIN_HINTS = ("facetime", "built-in", "builtin", "macbook")
CAMO_HINTS = ("camo",)
LIVE_KINDS = ("builtin", "camo")


# ----------------------------------------------------------------------------
# Listing cameras by name (macOS)
# ----------------------------------------------------------------------------
def _devices_pyobjc():
    """[(name, uniqueID)] from AVFoundation itself (needs pyobjc-framework-AVFoundation)."""
    try:
        import AVFoundation  # noqa: F401  (optional dependency)
    except Exception:
        return None
    devs = list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo))
    devs += list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeMuxed))
    return [(str(d.localizedName()), str(d.uniqueID())) for d in devs]


def _parse_system_profiler(text):
    """Parse `system_profiler SPCameraDataType -json` -> [(name, uniqueID)]."""
    data = json.loads(text)
    out = []
    for cam in data.get("SPCameraDataType", []):
        name = cam.get("_name", "")
        uid = next((v for k, v in cam.items() if "unique" in k.lower()), None)
        if name:
            out.append((name, uid))
    return out


def _devices_system_profiler():
    try:
        p = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                           capture_output=True, text=True, timeout=8)
        return _parse_system_profiler(p.stdout) if p.returncode == 0 else None
    except Exception:
        return None


def list_cameras():
    """Cameras in OpenCV index order: [(index, name)], or None if names are unknown
    (not a Mac, or neither listing method worked)."""
    if sys.platform != "darwin":
        return None
    devs = _devices_pyobjc() or _devices_system_profiler()
    return order_like_opencv(devs) if devs else None


def order_like_opencv(devs):
    """OpenCV sorts the devices by uniqueID (NSString compare), then indexes them.
    Without uniqueIDs we cannot know the order, so we return the names unordered
    (index None) - still useful to say whether Camo exists at all."""
    if any(uid is None for _, uid in devs):
        return [(None, name) for name, _ in devs]
    return [(i, name) for i, (name, _) in enumerate(sorted(devs, key=lambda d: d[1]))]


def matches(name, kind):
    n = (name or "").lower()
    hints = CAMO_HINTS if kind == "camo" else BUILTIN_HINTS
    return any(h in n for h in hints)


# ----------------------------------------------------------------------------
# Choosing the source
# ----------------------------------------------------------------------------
class CameraNotFound(Exception):
    pass


def parse_source(text):
    """'builtin' / 'camo' / '2' / 'clip.mp4' -> ('builtin'|'camo', None) | ('index', 2) | ('file', path)."""
    t = str(text).strip()
    if t.lower() in LIVE_KINDS:
        return t.lower(), None
    if t.lower() in ("mac", "laptop", "facetime", "webcam"):
        return "builtin", None
    if t.lower() in ("phone", "android"):
        return "camo", None
    if t.lstrip("-").isdigit():
        return "index", int(t)
    return "file", t


def resolve(kind, saved=None, cameras=None):
    """Which OpenCV index to open for 'builtin' or 'camo'.

    saved:   {"index": int, "name": str} written by tools/find_cameras.py, or None
    cameras: list_cameras() result (passed in by tests)
    Returns (index, how) where `how` explains the choice for the terminal.
    Raises CameraNotFound with a message meant for the user.
    """
    if cameras is None:
        cameras = list_cameras()
    if cameras:
        named = [(i, n) for i, n in cameras if matches(n, kind)]
        if not named:
            names = ", ".join(n for _, n in cameras) or "none"
            if kind == "camo":
                raise CameraNotFound(
                    "Camo camera not found. Open Camo Studio on the Mac and connect the phone "
                    f"(USB is the most stable). Cameras the Mac can see: {names}.")
            raise CameraNotFound(f"No built-in camera found. Cameras the Mac can see: {names}.")
        if saved and any(i == saved.get("index") and n == saved.get("name") for i, n in cameras):
            return saved["index"], f"verified earlier with tools/find_cameras.py ('{saved['name']}')"
        idx, name = named[0]
        if idx is not None:
            return idx, f"'{name}' found by name"
        if saved and saved.get("index") is not None:
            return saved["index"], f"'{name}' is present; index {saved['index']} saved by tools/find_cameras.py"
        if kind == "builtin" and len(cameras) == 1:
            return 0, f"'{name}' is the only camera"
        if kind == "builtin":   # what SENSE always did; keeps working if the order is unknown
            return 0, f"'{name}' is present but its number is unknown: using index 0 (check with tools/find_cameras.py)"
        raise CameraNotFound(
            f"'{name}' is connected but its camera number is unknown. "
            "Run: python tools/find_cameras.py   (one time) to pick it.")
    # No names available (not a Mac, or listing failed): only a human-verified index is trusted
    if saved and saved.get("index") is not None:
        return saved["index"], "saved by tools/find_cameras.py (camera names unavailable)"
    if kind == "builtin":
        return 0, "camera names unavailable: using index 0, the usual default"
    raise CameraNotFound("Cannot tell which camera is Camo on this computer. "
                         "Run: python tools/find_cameras.py   (one time) to pick it.")


# ----------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------
def normalize_frame(frame, max_width):
    """Any frame -> 3-channel BGR uint8, no wider than max_width (aspect kept)."""
    if frame is None or frame.size == 0:
        return None
    if frame.dtype != np.uint8:
        frame = cv2.convertScaleAbs(frame)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    if max_width and w > max_width:
        frame = cv2.resize(frame, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    return frame


class FreezeDetector:
    """Notices a picture that stopped changing completely (Camo placeholder or a
    stalled stream). Compares tiny grey thumbnails, so it costs almost nothing."""
    THRESHOLD = 0.05      # mean absolute difference (0-255 grey levels); live video is far above
    FREEZE_S = 3.0

    def __init__(self):
        self.prev = None
        self.still_since = None

    def update(self, frame, now):
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32)
        if self.prev is not None and small.shape == self.prev.shape \
                and float(np.mean(np.abs(small - self.prev))) < self.THRESHOLD:
            self.still_since = self.still_since or now
        else:
            self.still_since = None
        self.prev = small
        return self.frozen(now)

    def frozen(self, now):
        return self.still_since is not None and now - self.still_since >= self.FREEZE_S

    def reset(self):
        self.prev, self.still_since = None, None


class CameraSource:
    """cv2.VideoCapture with source choice, frame normalisation, freeze detection
    and reconnect. Used only by the main thread (no extra threads).

    read() -> (ok, frame). For a video file, ok=False means the clip ended.
    For a live camera, ok=False means "no frame right now" and the caller should
    keep its loop (keys, voice) running; reconnect attempts happen inside read().
    """
    RETRY_S = 2.0
    FALLBACK_AFTER_S = 6.0     # Camo gone this long -> use the built-in camera
    REQUEST_SIZE = (1280, 720)

    def __init__(self, spec, max_width=1280, saved=None, log=None, announce=None, opener=None,
                 cameras=None, clock=time.time):
        self.kind, self.value = parse_source(spec)
        self.max_width = max_width
        self.saved = saved or {}          # {"camo": {"index", "name"}, "builtin": {...}}
        self.log = log or (lambda msg: print(msg))
        self.announce = announce or (lambda msg: None)
        self.opener = opener or (lambda src: cv2.VideoCapture(src))   # looked up at call time (smoke_run patches it)
        self.cameras = cameras
        self.clock = clock
        self.cap = None
        self.index = None
        self.description = ""
        self.lost_since = None
        self.next_retry = 0.0
        self.freeze = FreezeDetector()
        self.frozen_reported = False
        self.size_changed = False        # tells the main loop to reset optical flow / tracker
        self._last_shape = None
        self.fallback_from = None

    @property
    def is_file(self):
        return self.kind == "file"

    @property
    def is_live(self):
        return not self.is_file

    def label(self):
        return {"camo": "phone camera (Camo)", "builtin": "built-in camera"}.get(
            self.kind, f"camera {self.index}" if self.kind == "index" else self.value)

    # --- opening -------------------------------------------------------------
    def _target(self):
        if self.kind == "file":
            return self.value, f"video file {self.value}"
        if self.kind == "index":
            return self.value, f"camera index {self.value} (given on the command line)"
        idx, how = resolve(self.kind, self.saved.get(self.kind), self.cameras)
        return idx, f"{self.label()} = index {idx}: {how}"

    def _open_once(self):
        target, how = self._target()          # may raise CameraNotFound
        cap = self.opener(target)
        if self.is_live:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.REQUEST_SIZE[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.REQUEST_SIZE[1])
            except Exception:
                pass
        if not cap.isOpened():
            cap.release()
            raise CameraNotFound(f"Could not open {how}.")
        ok, frame = False, None
        for _ in range(30 if self.is_live else 1):   # cameras often return a few empty frames first
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                break
        if not ok or frame is None or not frame.size:
            cap.release()
            raise CameraNotFound(f"Opened {how}, but no picture arrived.")
        self.cap, self.index, self.description = cap, target, how
        self.freeze.reset()
        self.frozen_reported = False
        return normalize_frame(frame, self.max_width)

    def open(self):
        """First open at start-up. Falls back from Camo to the built-in camera
        with a clear message. Returns the first frame or raises CameraNotFound."""
        try:
            frame = self._open_once()
        except CameraNotFound as e:
            if self.kind != "camo":
                raise
            self.log(f"📷 {e}")
            self.announce("I can't find the phone camera. Using the laptop camera. "
                          "Open Camo Studio and press C to switch.")
            self.fallback_from, self.kind = "camo", "builtin"
            frame = self._open_once()
        self.log(f"📷 Camera: {self.description}. Frames {frame.shape[1]}x{frame.shape[0]}"
                 + (f" (anything wider than {self.max_width} is scaled down)" if self.max_width else ""))
        self._last_shape = frame.shape
        return frame

    def switch(self, kind):
        """Switch between 'builtin' and 'camo' at run time (key C). Returns a
        sentence for the user. On failure the old camera keeps running."""
        old = (self.kind, self.value, self.cap, self.index, self.description)
        self.kind, self.value = kind, None
        try:
            new_cap_frame = self._open_once()
        except CameraNotFound as e:
            self.kind, self.value, self.cap, self.index, self.description = old
            self.log(f"📷 Switch failed: {e}")
            return (f"I can't use the {'phone' if kind == 'camo' else 'laptop'} camera. "
                    + ("Check that Camo Studio is open and the phone is connected." if kind == "camo" else ""))
        if old[2] is not None and old[2] is not self.cap:
            old[2].release()
        self.fallback_from = None
        self.lost_since = None
        self.size_changed = True
        self._last_shape = new_cap_frame.shape
        self.log(f"📷 Switched: {self.description}")
        return f"Switched to the {self.label()}."

    def other_kind(self):
        return "builtin" if self.kind == "camo" else "camo"

    # --- reading -------------------------------------------------------------
    def read(self):
        now = self.clock()
        if self.cap is not None:
            ok, frame = self.cap.read()
            frame = normalize_frame(frame, self.max_width) if ok else None
            if frame is not None:
                if self.lost_since is not None:
                    self.log(f"📷 Camera back after {now - self.lost_since:.1f}s")
                    self.announce("Camera is back.")
                    self.lost_since = None
                if self._last_shape is not None and frame.shape != self._last_shape:
                    self.size_changed = True
                self._last_shape = frame.shape
                if self.is_live:
                    self._check_freeze(frame, now)
                return True, frame
            if self.is_file:
                return False, None
            # live camera stopped giving frames
            self.cap.release()
            self.cap = None
            self.lost_since = now
            self.next_retry = now + self.RETRY_S
            self.log(f"📷 Camera lost ({self.description}). Retrying every {self.RETRY_S:.0f}s...")
            self.announce("Camera lost. "
                          + ("Check the phone and Camo." if self.kind == "camo" else "Trying again."))
            return False, None
        if self.is_file:
            return False, None
        # reconnecting (one attempt per RETRY_S, never a busy loop)
        if now >= self.next_retry:
            self.next_retry = now + self.RETRY_S
            try:
                frame = self._open_once()
                self.size_changed = self._last_shape is not None and frame.shape != self._last_shape
                self._last_shape = frame.shape
                self.log(f"📷 Camera back after {now - self.lost_since:.1f}s: {self.description}")
                self.announce("Camera is back.")
                self.lost_since = None
                return True, frame
            except CameraNotFound as e:
                self.log(f"📷 Still no camera: {e}")
                if self.kind == "camo" and now - self.lost_since >= self.FALLBACK_AFTER_S:
                    msg = self.switch("builtin")
                    if self.kind == "builtin":
                        self.fallback_from = "camo"
                        self.announce("The phone camera is not coming back. Using the laptop camera. "
                                      "Press C to try the phone again.")
                    else:
                        self.log(f"📷 {msg}")
        return False, None

    def _check_freeze(self, frame, now):
        if self.freeze.update(frame, now):
            if not self.frozen_reported:
                self.frozen_reported = True
                self.log(f"📷 The picture has not changed for {FreezeDetector.FREEZE_S:.0f}s ({self.description}).")
                self.announce("The camera picture is frozen. "
                              + ("Check that the phone is connected in Camo." if self.kind == "camo"
                                 else "Check the camera."))
        elif self.frozen_reported and self.freeze.still_since is None:
            self.frozen_reported = False
            self.log("📷 Picture moving again.")

    def take_size_changed(self):
        changed, self.size_changed = self.size_changed, False
        return changed

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
