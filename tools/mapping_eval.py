"""
tools/mapping_eval.py - how well does SENSE recognise YOUR rooms? (Phase 1)

    python tools/mapping_eval.py TEACH_DIR QUERY_DIR [--checks 3]

TEACH_DIR has one folder per room with the frames you taught it from, e.g.
the frames of a slow turn (record with the Mac camera, export every 10th
frame).  QUERY_DIR has one folder per room with OTHER frames of that room
(another day, other light, another spot), and optionally a folder called
"_unknown" with places you never taught (they should NOT be called a room).

For every query image, SENSE's real recogniser (mapping/recognise.py, same
settings as the app) is run a few times on it, like the filter does live, and
the spoken answer is classified:

    right       "You are in the kitchen" for a kitchen frame
    unsure      "I think..." / "I'm not sure" (honest, not wrong)
    WRONG       a confident answer naming another room, or a room for an
                untaught place

The model file must already be in models/ (run SENSE once, or pass --download).
Nothing is written to sense_memory.db.
"""
import argparse
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mapping.config import MappingConfig                    # noqa: E402
from mapping.fingerprint import Embedder, orb_features      # noqa: E402
from mapping.recognise import UNKNOWN, Recogniser, RoomModel  # noqa: E402

EXT = (".jpg", ".jpeg", ".png", ".bmp")


def images(folder):
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(EXT):
            img = cv2.imread(os.path.join(folder, f))
            if img is not None:
                yield f, img


def features(embedder, img):
    pts, des = orb_features(img)
    return {"emb": embedder.embed(img) if embedder else None, "pts": pts, "des": des, "landmarks": []}


def evaluate(teach_dir, query_dir, checks=3, embedder=None, cfg=None, verbose=True):
    cfg = cfg or MappingConfig()
    models, ids = [], {}
    for i, room in enumerate(sorted(d for d in os.listdir(teach_dir) if os.path.isdir(os.path.join(teach_dir, d)))):
        views = [features(embedder, img) for _, img in images(os.path.join(teach_dir, room))]
        models.append(RoomModel(i, room, views))
        ids[room] = i
    counts = {"right": 0, "right (I think)": 0, "unsure": 0, "hedged wrong": 0, "wrong": 0}
    rows = []
    for room in sorted(os.listdir(query_dir)):
        folder = os.path.join(query_dir, room)
        if not os.path.isdir(folder):
            continue
        for fname, img in images(folder):
            rec = Recogniser(cfg)
            rec.set_rooms(models)
            f = features(embedder, img)
            for k in range(checks):
                rec.update(f["emb"], f["pts"], f["des"], set(), k)
            state, p = rec.best()
            said = UNKNOWN if state == UNKNOWN else models[state].name
            named = state != UNKNOWN and p >= cfg.likely_prob
            sure = named and p >= cfg.sure_prob and rec.verified(state)
            truth = None if room == "_unknown" else room
            if named and said == truth:
                verdict = "right" if sure else "right (I think)"
            elif sure:
                verdict = "wrong"
            elif named:
                verdict = "hedged wrong"     # "I think you are in X", X wrong
            else:
                verdict = "unsure" if truth is not None else "right"   # untaught place: not naming it is right
            counts[verdict] += 1
            rows.append((room, fname, said, p, verdict))
            if verbose:
                print(f"{verdict.upper():15} {room}/{fname}: {said} p={p:.2f} "
                      f"sims={ {models[k].name: round(v, 2) for k, v in rec.last_scores['sims'].items()} if len(models) <= 4 else ''}")
    n = sum(counts.values()) or 1
    print(f"\n{n} query images: {counts['right']} right and sure, {counts['right (I think)']} right "
          f"but hedged (\"I think\"), {counts['unsure']} not sure, {counts['hedged wrong']} wrong but "
          f"hedged, {counts['wrong']} WRONG and sure")
    return counts, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("teach_dir")
    ap.add_argument("query_dir")
    ap.add_argument("--checks", type=int, default=3)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--no-model", action="store_true", help="corner features only (no image model)")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    emb = None if a.no_model else Embedder.create(download=a.download)
    evaluate(a.teach_dir, a.query_dir, a.checks, emb, verbose=not a.quiet)


if __name__ == "__main__":
    main()
