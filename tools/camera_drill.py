"""
tools/camera_drill.py - the REAL sense_master.py loop with two fake LIVE cameras
(built-in + Camo) that get switched, unplugged and lost, headless.

    python tools/camera_drill.py CLIPS_DIR [--json out.json]

CLIPS_DIR is the folder from tools/make_test_videos.py. The drill makes
camo_1080p.mp4 from approach_far.mp4: 1920x1080 with hand-held shake, like a
phone through Camo. The "Mac camera" plays pan_near.mp4 at 1280x720.
Cameras are found the way SENSE finds them on a Mac (by name, in OpenCV's
uniqueID order): index 0 = "Camo", index 1 = "FaceTime HD Camera".

Timeline (seconds after start):
    0   start with --camera camo (1080p frames must come out 1280x720)
    5   press C  -> built-in camera
    9   press C  -> Camo again
    12  "what do you see"  (YOLO + ByteTrack on the shaking Camo picture)
    14  Camo unplugged for 3 s -> "Camera lost", then "Camera is back"
    22  Camo gone for good -> after ~6 s "Using the laptop camera"
    34  "what do you see"
    36  Q
Checks are printed as PASS/FAIL; exit code 1 on any failure.
"""
import argparse
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("clips")
ap.add_argument("--json")
a = ap.parse_args()

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def make_camo_clip(src, dst):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    rng = np.random.default_rng(3)
    dx = dy = ang = 0.0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        # hand-held phone: a slow random walk of position and roll
        dx = 0.85 * dx + rng.normal(0, 6)
        dy = 0.85 * dy + rng.normal(0, 4)
        ang = 0.9 * ang + rng.normal(0, 0.4)
        m = cv2.getRotationMatrix2D((960, 540), ang, 1.04)
        m[:, 2] += (dx, dy)
        out.write(cv2.warpAffine(f, m, (1920, 1080), borderMode=cv2.BORDER_REFLECT))
    out.release()


CAMO_CLIP = os.path.join(a.clips, "camo_1080p.mp4")
MAC_CLIP = os.path.join(a.clips, "pan_near.mp4")
if not os.path.exists(CAMO_CLIP):
    make_camo_clip(os.path.join(a.clips, "approach_far.mp4"), CAMO_CLIP)

spoken, logs = [], []
state = {"t0": None, "error": None, "max_threads": 0, "frames": 0, "shapes": set(), "opens": []}


def now():
    return 0.0 if state["t0"] is None else time.time() - state["t0"]


def camo_available():
    t = now()
    return not (14.0 <= t < 17.0 or t >= 22.0)


# --- hardware stand-ins -------------------------------------------------------
sr = types.ModuleType("speech_recognition")
sr.Recognizer = type("Recognizer", (), {"pause_threshold": 1.0, "energy_threshold": 300})
sr.Microphone = type("Microphone", (), {"list_microphone_names": staticmethod(lambda: [])})
sr.WaitTimeoutError = TimeoutError
sys.modules["speech_recognition"] = sr


class _Proc:
    def poll(self):
        return 0


_real_popen, _real_run = subprocess.Popen, subprocess.run
subprocess.Popen = lambda cmd, *x, **k: (spoken.append((round(now(), 1), cmd[-1])) or _Proc()) \
    if cmd and cmd[0] == "say" else _real_popen(cmd, *x, **k)
subprocess.run = lambda cmd, *x, **k: subprocess.CompletedProcess(cmd, 0) \
    if cmd and cmd[0] in ("say", "killall", "afplay") else _real_run(cmd, *x, **k)

_RealCapture = cv2.VideoCapture


class FakeLive:
    """A live camera: plays a clip in a loop at its frame rate; index 0 (Camo)
    disappears according to camo_available()."""
    def __init__(self, index, *x):
        self.index = index
        state["opens"].append((round(now(), 1), index))
        self.ok = index in (0, 1) and (index != 0 or camo_available())
        self.cap = _RealCapture(CAMO_CLIP if index == 0 else MAC_CLIP) if self.ok else None
        self.next_t = time.perf_counter()

    def isOpened(self):
        return self.ok

    def set(self, *x):
        return True

    def get(self, *x):
        return 0

    def read(self):
        if not self.ok or (self.index == 0 and not camo_available()):
            return False, None
        wait = self.next_t - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        self.next_t = max(self.next_t, time.perf_counter() - 0.2) + 1 / 15.0
        ok, f = self.cap.read()
        if not ok:
            self.cap.release()
            self.cap = _RealCapture(CAMO_CLIP if self.index == 0 else MAC_CLIP)
            ok, f = self.cap.read()
        return ok, f

    def release(self):
        if self.cap is not None:
            self.cap.release()


cv2.VideoCapture = FakeLive
cv2.imshow = lambda *x, **k: None
cv2.namedWindow = lambda *x, **k: None
cv2.setWindowProperty = lambda *x, **k: None

KEYS = [(5.0, "c"), (9.0, "c"), (36.0, "q")]


def wait_key(delay=1):
    if state["t0"] is None:
        state["t0"] = time.time()
    if delay and delay > 1:
        time.sleep(delay / 1000.0)
    state["frames"] += 1
    state["max_threads"] = max(state["max_threads"], threading.active_count())
    g = sys._getframe(1).f_globals
    if "frame" in g and getattr(g.get("frame"), "shape", None) is not None:
        state["shapes"].add(tuple(g["frame"].shape))
    if KEYS and now() >= KEYS[0][0]:
        k = KEYS.pop(0)[1]
        spoken.append((round(now(), 1), f"[key {k}]"))
        return ord(k)
    return -1


cv2.waitKey = wait_key

import camera_source  # noqa: E402
camera_source.list_cameras = lambda: [(0, "Camo"), (1, "FaceTime HD Camera")]

import queue  # noqa: E402
QUESTIONS = [(12.0, "what do you see"), (34.0, "what do you see")]


class _Q(queue.Queue):
    def get_nowait(self):
        if QUESTIONS and now() >= QUESTIONS[0][0]:
            q = QUESTIONS.pop(0)[1]
            spoken.append((round(now(), 1), f"[asked: {q}]"))
            return ("text", q)
        return super().get_nowait()


queue.Queue = _Q

_print = print


def log_print(*x, **k):
    logs.append((round(now(), 1), " ".join(str(v) for v in x)))
    _print(*x, **k)


import builtins  # noqa: E402
builtins.print = log_print

import memory_db  # noqa: E402
tmp = tempfile.mkdtemp()
if os.path.exists(os.path.join(ROOT, "sense_memory.db")):
    shutil.copy(os.path.join(ROOT, "sense_memory.db"), os.path.join(tmp, "sense_memory.db"))
memory_db.DB_PATH = os.path.join(tmp, "sense_memory.db")
os.environ["SENSE_NO_TERMINAL_KEYS"] = "1"
os.environ["SENSE_SKIP_WHISPER_PRELOAD"] = "1"
os.chdir(ROOT)
sys.argv = ["sense_master.py", "--camera", "camo"]
try:
    runpy.run_path(os.path.join(ROOT, "sense_master.py"), run_name="__main__")
except SystemExit as e:
    if e.code not in (None, 0):
        state["error"] = f"SystemExit: {e.code}"
except Exception:  # noqa: BLE001
    state["error"] = traceback.format_exc()
builtins.print = _print
shutil.rmtree(tmp, ignore_errors=True)

# --- checks -------------------------------------------------------------------
said = [t for _, t in spoken]
log_text = [t for _, t in logs]


def first(pattern, items):
    return next(((t, s) for t, s in items if pattern.lower() in s.lower()), None)


checks = []


def check(name, ok, info=""):
    checks.append((name, bool(ok), info))


check("no crash, ran to the end", state["error"] is None and now() >= 35, state["error"] or f"{now():.0f}s")
check("Camo found by name and opened (index 0)", any("phone camera (Camo) = index 0" in s for s in log_text))
check("all frames given to YOLO are 3-channel and at most 1280 wide",
      state["shapes"] and all(len(s) == 3 and s[2] == 3 and s[1] <= 1280 for s in state["shapes"]),
      str(sorted(state["shapes"])))
check("1080p Camo frames scaled to 1280x720", (720, 1280, 3) in state["shapes"])
check("C switches to the built-in camera", first("Switched to the built-in camera", spoken))
check("C switches back to Camo", first("Switched to the phone camera", spoken))
resets = [t for t, s in logs if "motion tracking restarted" in s]
check("optical flow / tracker reset on each switch", len(resets) >= 2, str(resets))
lost, back = first("Camera lost", spoken), first("Camera is back", spoken)
check("unplugged Camo: 'Camera lost' then 'Camera is back'", lost and back and lost[0] < back[0] < 22,
      f"lost {lost}, back {back}")
fb = first("Using the laptop camera", spoken)
check("Camo gone for good: falls back to the laptop camera", fb and 26 <= fb[0] <= 33, str(fb))
ans = [s for t, s in spoken if s.startswith("Currently") or "I do not see" in s]
check("YOLO answers 'what do you see' on the shaking Camo feed",
      len(ans) >= 1 and ("dog" in ans[0].lower() or "bicycle" in ans[0].lower()), str(ans[:1]))
check("YOLO answers again after the fallback", len(ans) >= 2, str(ans[1:2]))
false_loom = [t for t, s in spoken if s.startswith("Stop. Something") and any(abs(t - r) < 1.5 for r in resets)]
check("no false 'Stop, something in front' right after a switch", not false_loom, str(false_loom))
check("never more than 2 Python threads (main + voice)", state["max_threads"] <= 2, str(state["max_threads"]))
retries = [t for t, i in state["opens"] if i == 0 and 14 <= t < 17]
check("reconnect does not spin (one open attempt per ~2 s)", 0 < len(retries) <= 3, str(retries))

print("\n==== SENSE said ====")
for t, s in spoken:
    print(f" [{t:5.1f}s] {s}")
print("\n==== camera drill ====")
for name, ok, info in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({info})" if info and not ok else ""))
failed = sum(not ok for _, ok, _ in checks)
print(f"{failed} failed.")
if a.json:
    with open(a.json, "w") as f:
        json.dump({"spoken": spoken, "logs": logs, "checks": checks, "error": state["error"]}, f)
sys.exit(1 if failed else 0)
