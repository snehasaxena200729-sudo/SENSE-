"""
tools/personal_e2e.py - the REAL sense_master.py loop (YOLO + ByteTrack + memory
+ answers) on synthetic clips: "this is my cup", turn 45 degrees to another
cup, "where is my cup?", turn back, restart SENSE, ask again.

    python tools/personal_e2e.py            # builds the clips, runs, checks
    python tools/personal_e2e.py --keep DIR # keep the clips and databases

The clips are a wide drawn "room" (textured wall, table, two drawn mugs that
YOLO detects as "cup" with ~0.85 confidence). The camera view slides across
it like a head turn; SENSE measures the turn from the image as usual. 45
degrees = 886 px at SENSE's assumed 65 degree view over 1280 px.

Two rooms:
  same      two IDENTICAL white cups (the hardest case)
  different a blue cup (yours) and a red cup

Run 1  teach cup A, turn to cup B, ask, turn back, ask
Run 2  restart (same database), start facing cup B, ask

Uses tools/smoke_run.py, which swaps only the camera, mic and `say` for the
clip, scripted questions and a transcript. Takes a few minutes on a CPU.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, FPS = 1280, 720, 15
TURN_PX = 886                      # 45 degrees


def draw_cup(img, cx, base, color):
    for x in range(cx - 70, cx + 70):
        shade = 0.6 + 0.4 * np.cos((x - cx) / 70 * 1.3)
        cv2.line(img, (x, base - 170), (x, base), tuple(int(v * shade) for v in color), 1)
    cv2.ellipse(img, (cx, base - 170), (70, 18), 0, 0, 360, tuple(int(v * 0.5) for v in color), -1)
    cv2.ellipse(img, (cx, base), (70, 18), 0, 0, 180, tuple(int(v * 0.7) for v in color), -1)
    cv2.ellipse(img, (cx + 70, base - 90), (40, 50), 0, -90, 90, tuple(int(v * 0.8) for v in color), 14)


def room(color_a, color_b, seed=0):
    rng = np.random.default_rng(seed)
    pano = np.full((H, W + TURN_PX + 40, 3), (170, 185, 200), np.uint8)
    for _ in range(260):               # posters / texture so the turn can be measured
        x, y = int(rng.integers(0, pano.shape[1])), int(rng.integers(0, 250))
        w, h = int(rng.integers(15, 90)), int(rng.integers(15, 90))
        c = tuple(int(v) for v in rng.integers(60, 230, 3))
        cv2.rectangle(pano, (x, y), (x + w, min(y + h, 250)), c, -1)   # kept above the cups
    cv2.rectangle(pano, (0, 470), (pano.shape[1], H), (60, 90, 130), -1)          # table top
    for _ in range(400):
        x, y = int(rng.integers(0, pano.shape[1])), int(rng.integers(480, H))
        cv2.circle(pano, (x, y), int(rng.integers(2, 6)), (45, 75, 110), -1)        # wood grain
    draw_cup(pano, W // 2, 480, color_a)                       # cup A: centre of view 0
    draw_cup(pano, W // 2 + TURN_PX, 480, color_b)             # cup B: 45 degrees right
    return pano


def write_clip(path, pano, plan, seed=1):
    """plan: list of (frames, start_offset_px, end_offset_px)."""
    rng = np.random.default_rng(seed)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for n, a, b in plan:
        for k in range(n):
            off = int(round(a + (b - a) * (k + 1) / n))
            frame = pano[:, off:off + W].astype(np.int16)
            frame += rng.normal(0, 3, frame.shape).astype(np.int16)      # sensor noise
            out.write(np.clip(frame, 0, 255).astype(np.uint8))
    out.release()


def smoke(clip, frames, questions, db, out_json):
    cmd = [sys.executable, os.path.join(ROOT, "tools", "smoke_run.py"), clip, str(frames),
           *questions, "--db", db, "--json", out_json]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    with open(out_json) as f:
        run = json.load(f)
    run["stdout"] = r.stdout
    return run


def answer_to(run, question):
    sp = run["spoken"]
    for k, (_, t) in enumerate(sp):
        if t == f"[asked: {question}]":
            return sp[k + 1][1] if k + 1 < len(sp) else ""
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default=None)
    a = ap.parse_args()
    work = a.keep or tempfile.mkdtemp()
    os.makedirs(work, exist_ok=True)
    failures = []
    hold, turn = 75, 20
    plan1 = [(hold, 0, 0), (turn, 0, TURN_PX), (100, TURN_PX, TURN_PX), (turn, TURN_PX, 0), (85, 0, 0)]
    q_teach, q_away, q_back = "this is my cup@50", "where is my cup@180", "where is my cup@285"
    plan2 = [(110, TURN_PX, TURN_PX)]                   # restarted facing cup B
    q_restart = "where is my cup@80"

    for name, ca, cb in [("same", (235, 235, 235), (235, 235, 235)),
                         ("different", (200, 120, 40), (40, 40, 200))]:
        pano = room(ca, cb)
        c1, c2 = os.path.join(work, f"{name}_run1.mp4"), os.path.join(work, f"{name}_run2.mp4")
        write_clip(c1, pano, plan1)
        write_clip(c2, pano, plan2, seed=2)
        db = os.path.join(work, f"{name}.db")
        if os.path.exists(db):
            os.remove(db)
        open(db, "w").close()                           # a fresh, empty memory
        print(f"== {name}: run 1 (teach, turn 45 degrees, ask, turn back, ask)")
        r1 = smoke(c1, sum(p[0] for p in plan1), [q_teach, q_away, q_back], db, db + ".r1.json")
        print(f"== {name}: run 2 (restart facing the other cup, ask)")
        r2 = smoke(c2, sum(p[0] for p in plan2), [q_restart], db, db + ".r2.json")
        for run in (r1, r2):
            if run["error"]:
                failures.append(f"{name}: exception {run['error'].strip().splitlines()[-1]}")
        taught = answer_to(r1, "this is my cup")
        away = answer_to(r1, "where is my cup")
        sp = r1["spoken"]
        idx = [k for k, (_, t) in enumerate(sp) if t == "[asked: where is my cup]"]
        back = sp[idx[1] + 1][1] if len(idx) > 1 and idx[1] + 1 < len(sp) else None
        after = answer_to(r2, "where is my cup")
        print(f"   taught : {taught}\n   away   : {away}\n   back   : {back}\n   restart: {after}")
        if not taught or "memorized your cup" not in taught:
            failures.append(f"{name}: teaching failed: {taught!r}")
        for label, text in (("turned away", away), ("restarted facing cup B", after)):
            if text is None or re.search(r"\bI see your cup\b|\bI can see your cup\b", text):
                failures.append(f"{name}, {label}: claimed to see my cup: {text!r}")
            elif "can't see it" not in text and "looks like yours" not in text:
                failures.append(f"{name}, {label}: did not say it can't see it: {text!r}")
        expect = "looks like yours" if name == "same" else "doesn't look like yours"
        if away and expect not in away:
            failures.append(f"{name}, turned away: the other cup should be described ({expect!r}): {away!r}")
        if away and re.search(r"next to the \w+ cup", away):
            failures.append(f"{name}: my cup described next to a ghost of itself: {away!r}")
        if not back or "your cup" not in back or "I see your cup" not in back:
            failures.append(f"{name}, turned back: cup A not recognised again: {back!r}")
        con = __import__("sqlite3").connect(db)
        row = con.execute("SELECT world_angle, session_id FROM personal_instances WHERE name='my cup'").fetchone()
        mem = con.execute("SELECT world_angle FROM environmental_memory WHERE object_name='my cup'").fetchone()
        con.close()
        print(f"   stored : personal_instances angle {row and round(row[0], 1)}, "
              f"environmental_memory angle {mem and mem[0] and round(mem[0], 1)}")
        if row is None or row[0] is None or abs(row[0]) > 15:
            failures.append(f"{name}: stored place of my cup is not cup A's: {row}")
        if mem is None or mem[0] is None or abs(mem[0]) > 15:
            failures.append(f"{name}: environmental_memory 'my cup' is not at cup A: {mem}")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    for f in failures:
        print("FAIL:", f)
    print(f"RESULT: {'PASS' if not failures else 'FAIL'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
