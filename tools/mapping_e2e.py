"""
tools/mapping_e2e.py - Phase 1 rooms in the REAL sense_master.py loop
(camera loop, YOLO, heading estimate, voice path, SQLite), on clips made from
real indoor photos.

    python tools/mapping_e2e.py --pano PANO.jpg --strip A.jpg B.jpg C.jpg ... --unknown U.jpg [--keep DIR]

    --pano     an equirectangular 360-degree photo of a room   -> "kitchen"
    --strip    several photos of ANOTHER room, put side by side -> "bedroom"
    --unknown  a photo of a third place that is never taught

Each room becomes a wide strip covering 360 degrees at SENSE's assumed
65-degree view over 1280 px (7088 px), and the camera view slides across it
like a slow turn, so SENSE measures the turn from the image as usual.

Run 1  "this room is the kitchen" + turn once; walk (cut) to the bedroom,
       "this room is the bedroom" + turn once; "which rooms do you know".
Run 2  restart SENSE with the same database: stand in the kitchen (a view
       between the taught ones, darker light), in the bedroom, in the
       unknown place, and ask "where am I?" each time. Also "this is the
       kitchen" must still go to the existing naming command.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, FPS = 1280, 720, 15
FULL = int(round(1280 * 360 / 65.0))          # 360 degrees in strip pixels


def pano_strip(path):
    pano = cv2.imread(path)
    ph, pw = pano.shape[:2]
    band = pano[int(ph * 0.25):int(ph * 0.75)]                 # +-45 degrees around the horizon
    return cv2.resize(band, (FULL, H), interpolation=cv2.INTER_CUBIC)


def photo_strip(paths):
    tiles = [cv2.imread(p) for p in paths]
    tiles = [cv2.resize(t, (int(t.shape[1] * H / t.shape[0]), H)) for t in tiles if t is not None]
    strip = np.hstack(tiles)
    reps = int(np.ceil(FULL / strip.shape[1]))
    return np.hstack([strip] * reps)[:, :FULL]


def view(strip, off, gamma=1.0):
    wide = np.hstack([strip, strip[:, :W]])                    # wrap round
    f = wide[:, int(off) % FULL:int(off) % FULL + W]
    if gamma != 1.0:
        f = np.clip(255 * (f / 255.0) ** gamma, 0, 255).astype(np.uint8)
    return f


def write(path, segments, seed=1):
    """segments: (frames, strip, start_px, end_px, gamma)"""
    rng = np.random.default_rng(seed)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = 0
    for frames, strip, a, b, g in segments:
        for k in range(frames):
            off = a + (b - a) * (k + 1) / frames
            f = view(strip, off, g).astype(np.int16) + rng.normal(0, 3, (H, W, 3)).astype(np.int16)
            out.write(np.clip(f, 0, 255).astype(np.uint8))
            n += 1
    out.release()
    return n


def smoke(clip, frames, questions, db, out_json):
    cmd = [sys.executable, os.path.join(ROOT, "tools", "smoke_run.py"), clip, str(frames),
           *questions, "--db", db, "--json", out_json]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    with open(out_json) as f:
        run = json.load(f)
    run["stdout"] = r.stdout
    return run


def answers(run):
    """[(question, next spoken line)] in order."""
    sp, out = run["spoken"], []
    for k, (_, t) in enumerate(sp):
        if t.startswith("[asked: "):
            nxt = next((s for _, s in sp[k + 1:] if not s.startswith("[asked: ")), "")
            out.append((t[8:-1], nxt))
    return out


def spoken_after(run, frame):
    return [t for f, t in run["spoken"] if f >= frame and not t.startswith("[asked")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pano", required=True)
    ap.add_argument("--strip", nargs="+", required=True)
    ap.add_argument("--unknown", required=True)
    ap.add_argument("--keep", default=None)
    a = ap.parse_args()
    work = a.keep or tempfile.mkdtemp()
    os.makedirs(work, exist_ok=True)
    kitchen, bedroom = pano_strip(a.pano), photo_strip(a.strip)
    unk = cv2.imread(a.unknown)
    unk_strip = np.hstack([cv2.resize(unk, (int(unk.shape[1] * H / unk.shape[0]), H))] * 10)[:, :FULL]
    turn_frames = 300                      # 360 degrees in 20 s = 18 degrees a second
    run1 = [(45, kitchen, 0, 0, 1.0), (turn_frames, kitchen, 0, FULL + 400, 1.0), (60, kitchen, FULL + 400, FULL + 400, 1.0),
            (45, bedroom, 0, 0, 1.0), (turn_frames, bedroom, 0, FULL + 400, 1.0), (75, bedroom, FULL + 400, FULL + 400, 1.0)]
    q1 = ["this room is the kitchen@20", "this room is the bedroom@410", "which rooms do you know@760"]
    k_off = FULL * 95 / 360.0              # 95 degrees: between two taught views
    b_off = FULL * 200 / 360.0
    run2 = [(90, kitchen, k_off, k_off, 1.5), (90, bedroom, b_off, b_off, 1.0), (90, unk_strip, 0, 0, 1.0),
            (60, kitchen, k_off + 300, k_off + 300, 1.0)]
    q2 = ["where am i@70", "where am i@160", "where am i@250", "this is the kitchen@300", "where am i@315"]
    c1, c2 = os.path.join(work, "rooms_run1.mp4"), os.path.join(work, "rooms_run2.mp4")
    n1, n2 = write(c1, run1), write(c2, run2, seed=2)
    db = os.path.join(work, "rooms.db")
    if os.path.exists(db):
        os.remove(db)
    shutil.copy(os.path.join(ROOT, "sense_memory.db"), db)      # a copy of the real memory
    print("== run 1: teach the kitchen and the bedroom by turning round")
    r1 = smoke(c1, n1, q1, db, db + ".r1.json")
    print("== run 2: restart, ask where am I in each place")
    r2 = smoke(c2, n2, q2, db, db + ".r2.json")
    fails = []
    for run in (r1, r2):
        if run.get("error"):
            fails.append("exception: " + run["error"].strip().splitlines()[-1])
    a1, a2 = answers(r1), answers(r2)
    for q, t in a1 + a2:
        print(f"   {q!r:32} -> {t}")
    said1 = " | ".join(t for _, t in r1["spoken"])
    print("   run 1 also said: " + " | ".join(t for t in spoken_after(r1, 0) if "learn" in t.lower() or "Halfway" in t or "turn" in t.lower()))
    if "Done. I've learned the kitchen" not in said1:
        fails.append("kitchen was not learned")
    if "Done. I've learned the bedroom" not in said1:
        fails.append("bedroom was not learned")
    rooms = dict(a1).get("which rooms do you know", "")
    if "kitchen" not in rooms or "bedroom" not in rooms:
        fails.append(f"room list: {rooms!r}")
    wh = [t for q, t in a2 if q == "where am i"]
    expect = ["kitchen", "bedroom", None, "kitchen"]
    for got, want in zip(wh, expect):
        if want is None:
            if got.startswith("You are in"):
                fails.append(f"untaught place named confidently: {got!r}")
        elif want not in got or "not sure" in got and "think" not in got:
            fails.append(f"wanted {want}: {got!r}")
    learn = dict(a2).get("this is the kitchen", "")
    if "room" in learn.lower() and ("turn slowly" in learn or "you are in" in learn.lower()):
        fails.append(f"'this is the kitchen' was taken by the home map: {learn!r}")
    print("\nPASS" if not fails else "\nFAIL\n  " + "\n  ".join(fails))
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
