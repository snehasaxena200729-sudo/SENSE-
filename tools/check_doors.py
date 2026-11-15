"""
tools/check_doors.py - does the door model actually see YOUR door?

Guessing thresholds does not fix door detection; measuring does. Take 5-10
photos of the door from where you would really stand (near and far, open and
closed, lights on and off) and run:

    python tools/check_doors.py photos/door_*.jpg
    python tools/check_doors.py photos/ --models yolov8n-oiv7.pt yolov8s-oiv7.pt

For each photo and each model it prints the BEST score for Door, Door handle
and Stairs, even below SENSE's thresholds (it asks the model down to 0.01),
plus what SENSE would decide with its current rules. --save-dir writes the
photos with the boxes drawn, so you can see WHAT the model calls a door.

How to read the result:
  * door scores >= 0.25 on most photos   -> the model is fine; check camera angle / framing
  * door 0.10-0.25 with a handle >= 0.30  -> SENSE now accepts it (handle evidence)
  * a bigger model scores much higher     -> switch SCENE_MODEL in sense_master.py
  * all models stay under 0.10            -> the door is outside what these models learned;
                                             fine-tune on ~100-200 labelled photos of your doors
The first run of a model downloads it once; after that it is offline.
"""
import argparse
import glob
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import hazards  # noqa: E402

WANTED = ("door", "door handle", "stairs")


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*")))
        else:
            files += sorted(glob.glob(p))
    return [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--models", nargs="+", default=["yolov8n-oiv7.pt"])
    ap.add_argument("--save-dir", help="write annotated copies here")
    args = ap.parse_args()

    from ultralytics import YOLO
    files = collect(args.photos)
    if not files:
        sys.exit("No photos found.")
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for model_name in args.models:
        model = YOLO(model_name)
        ids = {i: n.lower() for i, n in model.names.items() if n.lower() in WANTED}
        rules = hazards.SceneHazardDetector(model)
        print(f"\n=== {model_name} ({len(model.names)} classes) ===")
        print(f"{'photo':28} {'door':>6} {'handle':>7} {'stairs':>7}   SENSE decides")
        found = 0
        for path in files:
            img = cv2.imread(path)
            if img is None:
                continue
            res = model.predict(img, classes=list(ids), conf=0.01, imgsz=rules.IMG_SIZE, verbose=False)[0]
            raw = [(ids[int(b.cls[0])], float(b.conf[0]), tuple(int(v) for v in b.xyxy[0])) for b in res.boxes]
            best = {n: max([c for m, c, _ in raw if m == n], default=0.0) for n in WANTED}
            decided = rules.filter_detections(raw)
            names = sorted({n for n, _, _ in decided})
            found += "door" in names
            print(f"{os.path.basename(path)[:28]:28} {best['door']:6.2f} {best['door handle']:7.2f} "
                  f"{best['stairs']:7.2f}   {', '.join(names) or 'nothing'}")
            if args.save_dir:
                out = img.copy()
                for n, c, (x1, y1, x2, y2) in raw:
                    if c < 0.05:
                        continue
                    colour = (200, 0, 200) if (n, c, (x1, y1, x2, y2)) in decided else (128, 128, 128)
                    cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(out, f"{n} {c:.2f}", (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
                stem = os.path.splitext(os.path.basename(path))[0]
                cv2.imwrite(os.path.join(args.save_dir, f"{stem}_{os.path.splitext(model_name)[0]}.jpg"), out)
        print(f"Door accepted in {found} of {len(files)} photos.")


if __name__ == "__main__":
    main()
