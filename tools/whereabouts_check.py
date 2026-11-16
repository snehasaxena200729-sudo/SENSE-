"""
tools/whereabouts_check.py - "where did I leave it?" on real detector output,
including a restart. Runs the REAL sense_master.py loop twice through
tools/smoke_run.py, sharing one memory database between the runs.

    python tools/whereabouts_check.py PHOTO ITEM SURFACE [--landmark NAME] [--out DIR]

    e.g. python tools/whereabouts_check.py bedroom.jpg bottle bed --zoom 1.4 --centre 0.625 0.54

PHOTO must show ITEM resting on SURFACE (a table, bed or couch) in a way the
COCO YOLO model detects. The clips are synthetic (a still photo with sensor
noise, then the camera turns away to an empty wall), so this checks the logic
on real detections; it does not replace a test with the real camera.

Run 1  sees the item, turns away, asks "where is the ITEM?" and "find the ITEM".
Run 2  is a fresh start (new session) on an empty view with the SAME database:
       asks again, plus a question about something with no anchor.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, FPS = 1280, 720, 15


def make_clips(photo, out_dir, zoom=1.0, centre=(0.5, 0.5)):
    img = cv2.imread(photo)
    if img is None:
        raise SystemExit(f"cannot read {photo}")
    if zoom > 1.0:     # move closer: a 16:9 crop around `centre` (fractions of the photo)
        cw = img.shape[1] / zoom
        ch = min(img.shape[0], cw * 9 / 16)
        x = int(max(0, min(img.shape[1] - cw, centre[0] * img.shape[1] - cw / 2)))
        y = int(max(0, min(img.shape[0] - ch, centre[1] * img.shape[0] - ch / 2)))
        img = img[y:y + int(ch), x:x + int(cw)]
    s = max(W / img.shape[1], H / img.shape[0])
    img = cv2.resize(img, (int(img.shape[1] * s) + 1, int(img.shape[0] * s) + 1))
    y0 = (img.shape[0] - H) // 2
    x0 = (img.shape[1] - W) // 2
    view = img[y0:y0 + H, x0:x0 + W]
    rng = np.random.default_rng(7)
    wall = cv2.GaussianBlur(rng.normal(125, 30, (H, W * 3, 3)).clip(0, 255).astype(np.uint8), (7, 7), 0)
    strip = np.hstack([view, wall])                   # the room, then a plain wall to its right

    def write(path, frames):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for f in frames:
            vw.write(np.clip(f + rng.normal(0, 1.5, f.shape), 0, 255).astype(np.uint8))
        vw.release()

    def at(x):
        return strip[:, int(x):int(x) + W].astype(np.float32)

    run1 = [at(0)] * (8 * FPS)                                        # look at the item, still
    pan = 2 * FPS
    run1 += [at((W * 1.5) * (0.5 - 0.5 * np.cos(np.pi * i / pan))) for i in range(pan)]   # turn away
    run1 += [at(W * 1.5)] * (10 * FPS)                                # facing the wall
    run2 = [at(W * 1.5)] * (8 * FPS)                                  # restarted, facing the wall
    p1, p2 = os.path.join(out_dir, "where_run1.mp4"), os.path.join(out_dir, "where_run2.mp4")
    write(p1, run1)
    write(p2, run2)
    return p1, len(run1), p2, len(run2)


def run(video, frames, db, questions, out_json):
    cmd = [sys.executable, os.path.join(ROOT, "tools", "smoke_run.py"), video, str(frames),
           *[f"{q}@{f}" for q, f in questions], "--json", out_json, "--db", db]
    subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
    with open(out_json) as f:
        return json.load(f)


def answers_in(result):
    sp, out = result["spoken"], {}
    for k, (_, t) in enumerate(sp):
        m = re.match(r"\[asked: (.*)\]$", t)
        if m:
            out[m.group(1)] = sp[k + 1][1] if k + 1 < len(sp) else ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("item")
    ap.add_argument("surface")
    ap.add_argument("--landmark")
    ap.add_argument("--other", default="bed", help="something seen but never anchored (furniture)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--zoom", type=float, default=1.0, help="crop in, e.g. 1.4, when the item is small")
    ap.add_argument("--centre", type=float, nargs=2, default=(0.5, 0.5), metavar=("X", "Y"))
    a = ap.parse_args()
    out = a.out or tempfile.mkdtemp(prefix="whereabouts_")
    os.makedirs(out, exist_ok=True)
    p1, n1, p2, n2 = make_clips(a.photo, out, a.zoom, tuple(a.centre))
    db = os.path.join(out, "memory.db")
    open(db, "w").close()                 # start from an empty memory
    where, find = f"where is the {a.item}", f"find the {a.item}"
    r1 = run(p1, n1, db, [(where, n1 - 60), (find, n1 - 35), ("stop looking", n1 - 20)],
             os.path.join(out, "run1.json"))
    r2 = run(p2, n2, db, [(where, 40), (find, 60), ("stop looking", 75),
                          (f"where is the {a.other}", 90), ("where is my remote", 105)],
             os.path.join(out, "run2.json"))
    ans1, ans2 = answers_in(r1), answers_in(r2)
    place = f"on the {a.surface}"
    checks = [
        ("run 1 has no crash", r1["error"] is None),
        ("run 2 has no crash", r2["error"] is None),
        (f"run 1 '{where}' says '{place}'", place in ans1.get(where, "")),
        (f"run 1 '{find}' says '{place}'", place in ans1.get(find, "")),
        (f"after restart '{where}' says '{place}'", place in ans2.get(where, "")),
        ("after restart it does not sound certain",
         any(w in ans2.get(where, "") for w in ("restarted", "not sure", "may have"))),
        (f"after restart '{find}' says '{place}'", place in ans2.get(find, "")),
        ("no anchor: unknown item still gets the old answer",
         "no record" in ans2.get("where is my remote", "") or "did not catch" in ans2.get("where is my remote", "")),
    ]
    if a.landmark:
        checks.append((f"mentions the {a.landmark}", a.landmark in ans1.get(where, "")))
    if ans2.get(f"where is the {a.other}"):
        checks.append((f"'{a.other}' (no anchor) still answered, without anchor words",
                       place not in ans2[f"where is the {a.other}"]))
    con = __import__("sqlite3").connect(db)
    rows = con.execute("SELECT item_name, surface, landmark, relation, surface_votes, landmark_votes "
                       "FROM item_anchors").fetchall()
    print("item_anchors:", rows)
    for name, d in (("run 1", ans1), ("run 2", ans2)):
        for q, t in d.items():
            print(f"  {name}  Q: {q}\n          A: {t}")
    failed = 0
    for label, ok in checks:
        print(("PASS " if ok else "FAIL ") + label)
        failed += not ok
    print(f"{failed} failed, clips and logs in {out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
