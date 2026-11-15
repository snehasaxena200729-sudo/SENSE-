"""
tools/smoke_run.py - run the REAL sense_master.py loop on a video file, headless,
and print every sentence SENSE would speak. No camera, mic, window or `say` needed.

    python tools/smoke_run.py clip.mp4 [max_frames] ["question@frame" ...]

    e.g. python tools/smoke_run.py desk.mp4 200 "what do you see@60" "where is sneha@120"

Use it to check a change end to end before a field test: same YOLO, ByteTrack,
memory and notifier code as the live app. The SQLite memory is a temporary copy,
so your real sense_memory.db is not touched.
"""
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

video = sys.argv[1]
max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 300
questions = sorted((int(q.rsplit("@", 1)[1]), q.rsplit("@", 1)[0]) for q in sys.argv[3:])
spoken = []

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
        spoken.append(cmd[-1])
        return _FakeProc()
    return _real_popen(cmd, *a, **k)
def _run(cmd, *a, **k):
    if cmd and cmd[0] in ("say", "killall", "afplay"):
        return subprocess.CompletedProcess(cmd, 0)
    return _real_run(cmd, *a, **k)
subprocess.Popen, subprocess.run = _popen, _run

import cv2  # noqa: E402
frames = {"n": 0}
cv2.imshow = lambda *a, **k: None
def _wait(*a, **k):
    frames["n"] += 1
    return ord("q") if frames["n"] >= max_frames else -1
cv2.waitKey = _wait

# --- scripted voice questions, delivered like the listener thread would ----
import queue  # noqa: E402
class _ScriptedQueue(queue.Queue):
    def get_nowait(self):
        if questions and frames["n"] >= questions[0][0]:
            q = questions.pop(0)[1]
            spoken.append(f"[asked at frame {frames['n']}: {q}]")
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
try:
    runpy.run_path(os.path.join(ROOT, "sense_master.py"), run_name="__main__")
finally:
    print("\n==== SENSE would have said ====")
    for s in spoken:
        print(" -", s)
