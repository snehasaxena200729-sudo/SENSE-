"""
tools/smoke_run.py - run the REAL sense_master.py loop on a video file, headless,
and record everything SENSE would say. No camera, mic, window or `say` needed.

    python tools/smoke_run.py clip.mp4 [max_frames] ["question@frame" ...]
        [--json out.json] [--stats-every N] [--loop]

    e.g. python tools/smoke_run.py desk.mp4 200 "what do you see@60" "where is sneha@120"

Same YOLO, ByteTrack, face, memory, events and notifier code as the live app.
The SQLite memory is a temporary copy, so your real sense_memory.db is untouched.

--json writes: spoken sentences with frame numbers, any exception, per-frame
loop time, and (every N frames) process memory and the size of SENSE's
internal tables - used by tools/run_checks.py and the stress test.
--loop restarts the video when it ends (long stress runs from a short clip).
--fast reads frames as fast as possible. By default frames are paced to the
clip's frame rate, like a camera: SENSE's timing logic (gaps, trends, cooldowns)
works in seconds, and an unpaced run squeezes a 2 s gap into a fraction of that.
"""
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

args = sys.argv[1:]
opts = {"--json": None, "--stats-every": "0"}
flags = {"--loop": False, "--fast": False}
rest = []
i = 0
while i < len(args):
    if args[i] in opts:
        opts[args[i]] = args[i + 1]
        i += 2
    elif args[i] in flags:
        flags[args[i]] = True
        i += 1
    else:
        rest.append(args[i])
        i += 1
video = rest[0]
max_frames = int(rest[1]) if len(rest) > 1 else 300
questions = sorted((int(q.rsplit("@", 1)[1]), q.rsplit("@", 1)[0]) for q in rest[2:])
stats_every = int(opts["--stats-every"])
spoken = []          # (frame, text)
frame_times = []
samples = []
state = {"n": 0, "t_last": None, "error": None}


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 * 1024.0) if sys.platform == "darwin" else r / 1024.0


# --- stand-ins for hardware -------------------------------------------------
sr = types.ModuleType("speech_recognition")
class _Recognizer:
    pause_threshold = 1.0
    energy_threshold = 300
sr.Recognizer = _Recognizer
sr.Microphone = object
sr.WaitTimeoutError = TimeoutError
sys.modules["speech_recognition"] = sr

class _FakeProc:
    def poll(self):
        return 0
_real_popen, _real_run = subprocess.Popen, subprocess.run
def _popen(cmd, *a, **k):
    if cmd and cmd[0] == "say":
        spoken.append((state["n"], cmd[-1]))
        return _FakeProc()
    return _real_popen(cmd, *a, **k)
def _run(cmd, *a, **k):
    if cmd and cmd[0] in ("say", "killall", "afplay"):
        return subprocess.CompletedProcess(cmd, 0)
    return _real_run(cmd, *a, **k)
subprocess.Popen, subprocess.run = _popen, _run

import cv2  # noqa: E402

_RealCapture = cv2.VideoCapture
class _LoopingCapture:
    """Restarts the clip at its end when --loop is given."""
    def __init__(self, src, *a):
        self.src = src
        self.cap = _RealCapture(src, *a)
    def read(self):
        if not flags["--fast"]:
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
            self.t0 = getattr(self, "t0", None) or time.perf_counter()
            self.k = getattr(self, "k", 0) + 1
            wait = self.t0 + self.k / fps - time.perf_counter()
            if wait > 0:
                time.sleep(wait)       # never faster than a real camera
        ok, frame = self.cap.read()
        if not ok and flags["--loop"] and isinstance(self.src, str):
            self.cap.release()
            self.cap = _RealCapture(self.src)
            ok, frame = self.cap.read()
        return ok, frame
    def release(self):
        self.cap.release()
    def __getattr__(self, k):
        return getattr(self.cap, k)
cv2.VideoCapture = _LoopingCapture

TABLES = ["live_memory", "live_people", "track_votes", "track_conf", "track_locked_identities", "track_last_seen",
          "active_people_crops", "active_object_crops", "face_eye_px", "last_db_write"]

cv2.imshow = lambda *a, **k: None
def _wait(*a, **k):
    now = time.perf_counter()
    if state["t_last"] is not None:
        frame_times.append(now - state["t_last"])
    state["t_last"] = now
    state["n"] += 1
    if stats_every and state["n"] % stats_every == 0:
        g = sys._getframe(1).f_globals          # sense_master's globals
        sizes = {t: len(g[t]) for t in TABLES if t in g}
        for name, attr in (("person_gate.history", ("person_gate", "history")),
                           ("approach.history", ("approach", "history")),
                           ("coco_objects", ("coco_objects", "objects")),
                           ("scene_objects", ("scene_objects", "objects")),
                           ("resolver.tracks", ("resolver", "track_to_entity")),
                           ("resolver.entities", ("resolver", "entities")),
                           ("event_tracker.items", ("event_tracker", "items")),
                           ("notifier.pending", ("notifier", "pending")),
                           ("notifier.keys", ("notifier", "last_spoken_key_at"))):
            obj = g.get(attr[0])
            if obj is not None and hasattr(obj, attr[1]):
                sizes[name] = len(getattr(obj, attr[1]))
        objs = []
        for tr_name in ("coco_objects", "scene_objects"):
            tr = g.get(tr_name)
            if tr is None:
                continue
            for o in tr.objects.values():
                cd = o.current_distance(time.time())
                objs.append([tr_name[:5], o.oid, o.cls, None if cd is None else round(cd, 2),
                             bool(o.approaching(time.time())), o.armed, o.hits])
        samples.append({"frame": state["n"], "rss_mb": round(rss_mb(), 1), "sizes": sizes, "objects": objs,
                        "threads": __import__("threading").active_count()})
    return ord("q") if state["n"] >= max_frames else -1
cv2.waitKey = _wait

# --- scripted voice questions, delivered like the listener thread would ----
import queue  # noqa: E402
class _ScriptedQueue(queue.Queue):
    def get_nowait(self):
        if questions and state["n"] >= questions[0][0]:
            q = questions.pop(0)[1]
            spoken.append((state["n"], f"[asked: {q}]"))
            return ("text", q)
        return super().get_nowait()
queue.Queue = _ScriptedQueue

# --- temporary copy of the memory database ----------------------------------
import memory_db  # noqa: E402
tmp = tempfile.mkdtemp()
src_db = os.path.join(ROOT, "sense_memory.db")
if os.path.exists(src_db):
    shutil.copy(src_db, os.path.join(tmp, "sense_memory.db"))
memory_db.DB_PATH = os.path.join(tmp, "sense_memory.db")

os.chdir(ROOT)
sys.argv = ["sense_master.py", "--source", video]
t0 = time.time()
try:
    runpy.run_path(os.path.join(ROOT, "sense_master.py"), run_name="__main__")
except SystemExit:
    pass
except Exception:  # noqa: BLE001
    state["error"] = traceback.format_exc()
    print(state["error"])
finally:
    wall = time.time() - t0
    db_rows = {}
    try:
        import sqlite3
        con = sqlite3.connect(memory_db.DB_PATH)
        for t in ("environmental_memory", "events", "entities", "face_profiles"):
            db_rows[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        db_rows["integrity"] = con.execute("PRAGMA integrity_check").fetchone()[0]
        con.close()
    except Exception as e:  # noqa: BLE001
        db_rows["error"] = str(e)
    print("\n==== SENSE would have said ====")
    for f, s in spoken:
        print(f" - [{f}] {s}")
    if opts["--json"]:
        with open(opts["--json"], "w") as f:
            json.dump({"video": video, "frames": state["n"], "wall_s": wall, "error": state["error"],
                       "spoken": spoken, "frame_times": frame_times, "samples": samples,
                       "db_rows": db_rows}, f)
    shutil.rmtree(tmp, ignore_errors=True)
