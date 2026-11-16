"""
tools/door_distance_test.py - calibrate and tape-measure SENSE's door distances.

A distance you have not checked against a tape measure is a guess. This tool
runs the SAME door model, tracker and door_distance.py code as SENSE on your
camera and compares the result with distances you measured.

1) CALIBRATE (once per camera and door type, about a minute)
    python tools/door_distance_test.py --calibrate --distance-ft 6
   Stand facing the door, the laptop camera lens exactly 6 feet from the door
   (tape measure on the floor), door roughly in the middle of the picture.
   Press SPACE. It collects about 2 seconds of door boxes and saves
   "box size x distance" to sense_memory.db. sense_master.py uses it on the
   next start ("Door distance: calibrated."). Add --dry-run to only print it.
   Why this works: the camera only sees focal length x real door width. One
   measurement of that product cancels the focal-length guess, the door-size
   guess and the "box includes the door frame" bias together.

2) TEST at measured distances
    python tools/door_distance_test.py --test --marks 2 4 6 8 --label facing
   At each mark, press SPACE and hold still for ~2 seconds. Repeat with
   --label side (2 ft to the side of each mark), --label dark, --label dim,
   --label half-hidden, --label other-door ... Every run is appended to
   door_distance_results.csv and an error table is printed.
   Keys: SPACE capture, s skip this mark, q quit.

3) FROM PHOTOS (repeatable): name photos after the true distance,
   e.g. doors/6ft_facing.jpg, doors/4ft_side.jpg, doors/2.5m.jpg
    python tools/door_distance_test.py --images doors
   (single frames: no smoothing, so this shows the raw measurement error)

Reported per capture: true distance, SENSE's raw and smoothed estimate, the
cue used (height or width), whether SENSE would call it reliable, the error,
and exactly what SENSE would answer to "where is the door?".
"""
import argparse
import csv
import glob
import os
import re
import statistics
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import distance  # noqa: E402
import spatial  # noqa: E402
import door_distance as dd  # noqa: E402
from continuity import ObjectContinuity  # noqa: E402

FT = 0.3048
CAPTURE_S = 2.0


def settings():
    """(get, set) for sense_memory.db settings; harmless fallbacks if unavailable."""
    try:
        import memory_db
    except Exception as e:  # noqa: BLE001
        print(f"(settings unavailable: {e})")
        return (lambda k, d=None: d), None

    def get(key, default=None):
        try:
            return memory_db.get_setting(key, default)
        except Exception:  # noqa: BLE001  (fresh database: no settings table yet)
            return default

    def put(key, value):
        memory_db.init_database()          # creates the settings table if needed
        memory_db.set_setting(key, value)
    return get, put


def load_detector(model_path):
    from ultralytics import YOLO
    from hazards import SceneHazardDetector
    det = SceneHazardDetector(YOLO(model_path))
    if "door" not in det.class_ids.values():
        sys.exit(f"{model_path} has no Door class")
    return det


def doors_in(det, frame, now):
    """Accepted door boxes in this frame (the model runs on every call here)."""
    found = det.detect(frame, det.EVERY_N, now) or []
    return [(c, b) for n, c, b in found if n == "door"]


def parse_true_ft(name):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(ft|feet|m|cm)", name.lower())
    if not m:
        return None
    v, unit = float(m.group(1)), m.group(2)
    return {"ft": v, "feet": v, "m": v / FT, "cm": v / 100 / FT}[unit]


class Session:
    """The door half of sense_master: continuity + DoorRanger, nothing else."""

    def __init__(self, cal, f_px, focal_unc):
        self.objs = ObjectContinuity({"door"}, source="scene")
        self.ranger = dd.DoorRanger(cal)
        self.f_px, self.focal_unc = f_px, focal_unc
        self.raw = []          # (metres, cue) of every usable measurement
        self.said = []

    def feed(self, boxes, now, fw, fh):
        dets = [{"cls": "door", "box": b, "conf": c, "bt_id": None, "can_create": True, "est": None}
                for c, b in boxes]
        _, created = self.objs.update(dets, now, (0.0, 0.0), frame_size=(fw, fh))
        for obj, _ in created:
            self.ranger.on_created(obj, now, spatial.pixel_to_angle((obj.box[0] + obj.box[2]) / 2, fw))
        self.said += [t for _, t in self.ranger.update(self.objs.objects, now, fw, fh, self.f_px,
                                                      focal_unc=self.focal_unc)]
        if boxes:
            best = min(boxes, key=lambda cb: abs((cb[1][0] + cb[1][2]) / 2 - fw / 2))[1]
            m = dd.measure(best, fw, fh, self.f_px, self.focal_unc, self.ranger.cal)
            if m.ok:
                self.raw.append((m.metres, m.cue, m.unc))
            return best, m
        return None, None

    def smoothed(self, now):
        doors = self.ranger.visible_doors(now)
        if not doors:
            return None, False
        st = min(doors, key=lambda s: abs(s.angle))
        d = st.distance(now)
        return (d if d is not None else st.filter.metres(now)), d is not None


def draw(frame, box, m, msg):
    if box is not None:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 200), 2)
        txt = f"{m.metres / FT:.1f} ft ({m.cue}, est)" if m and m.ok else (m.reason if m else "")
        cv2.putText(frame, txt, (x1, max(20, y1 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
    cv2.putText(frame, msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def wait_key(cam, det, sess, msg):
    """Show the live view until SPACE / s / q. Returns the key."""
    while True:
        ok, frame = cam.read()
        if not ok:
            sys.exit("camera read failed")
        fh, fw = frame.shape[:2]
        box, m = sess.feed(doors_in(det, frame, time.time()), time.time(), fw, fh) if sess else (None, None)
        draw(frame, box, m, msg)
        cv2.imshow("SENSE door distance test", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord(" "), ord("s"), ord("q")):
            return chr(k)


def capture(cam, det, sess, seconds=CAPTURE_S):
    """Feed `seconds` of frames. Returns (boxes seen, frame size)."""
    end, boxes, size = time.time() + seconds, [], None
    while time.time() < end:
        ok, frame = cam.read()
        if not ok:
            break
        fh, fw = frame.shape[:2]
        size = (fw, fh)
        now = time.time()
        found = doors_in(det, frame, now)
        box, m = sess.feed(found, now, fw, fh)
        if box is not None:
            boxes.append(box)
        draw(frame, box, m, "hold still...")
        cv2.imshow("SENSE door distance test", frame)
        cv2.waitKey(1)
    return boxes, size


def focal(get, fw):
    try:
        ratio = float(get("focal_ratio", "0")) or None
    except ValueError:
        ratio = None
    unc = distance.CALIBRATED_FOCAL_UNCERTAINTY if ratio else distance.FOCAL_UNCERTAINTY
    return distance.focal_px(fw, focal_ratio=ratio), unc


def run_calibrate(args):
    get, put = settings()
    det = load_detector(args.scene_model)
    cam = cv2.VideoCapture(args.camera)
    ok, frame = cam.read()
    if not ok:
        sys.exit("cannot open the camera")
    fh, fw = frame.shape[:2]
    f_px, unc = focal(get, fw)
    sess = Session({}, f_px, unc)
    if wait_key(cam, det, sess, f"Stand {args.distance_ft:g} ft from the door, facing it. SPACE") != " ":
        return
    boxes, _ = capture(cam, det, sess)
    cam.release()
    cv2.destroyAllWindows()
    usable_w = [b for b in boxes if not distance.clipped_edges(b, fw, fh) & {"left", "right"}]
    usable_h = [b for b in boxes if not distance.clipped_edges(b, fw, fh) & {"top", "bottom"}]
    if len(usable_w) < 5:
        sys.exit(f"Only {len(usable_w)} usable door boxes (door must be fully in view left to right). "
                 "Move so the whole door width is in the picture and try again.")
    true_m = args.distance_ft * FT
    cal = {}
    for axis, bs, size in (("w", usable_w, lambda b: b[2] - b[0]), ("h", usable_h, lambda b: b[3] - b[1])):
        if len(bs) >= 5:
            b = sorted(bs, key=size)[len(bs) // 2]
            angle = spatial.pixel_to_angle((b[0] + b[2]) / 2, fw)
            cal[axis] = round(dd.calibration_value(size(b), true_m, angle, fw), 4)
    print(f"Door calibration from {len(boxes)} boxes at {args.distance_ft:g} ft: {cal}")
    if "w" in cal:
        print(f"  (uncalibrated SENSE would have assumed {f_px * dd.DOOR_WIDTH[0] / fw:.4f} for width)")
    if args.dry_run or put is None:
        print("Not saved (--dry-run).")
        return
    for axis, v in cal.items():
        put(f"door_cal_{axis}", v)
    print("Saved. sense_master.py will say 'Door distance: calibrated.' on the next start.")


def report(rows):
    print(f"\n{'true ft':>8} {'raw ft':>7} {'smooth':>7} {'error':>7} {'err %':>6}  {'cue':6} rel  label / SENSE says")
    errs = []
    for r in rows:
        sm = r["smoothed_ft"]
        e = (sm - r["true_ft"]) if sm != "" else None
        if e is not None and r["reliable"] == "yes":
            errs.append(abs(e) / r["true_ft"])
        raw = "-" if r["raw_ft"] == "" else r["raw_ft"]
        smooth = "-" if sm == "" else sm
        err = "-" if e is None else f"{e:+.1f}"
        pct = "-" if e is None else f"{100 * e / r['true_ft']:+.0f}"
        print(f"{r['true_ft']:8.1f} {raw:>7} {smooth:>7} {err:>7} {pct:>6}  "
              f"{r['cue']:6} {r['reliable']:3}  {r['label']} / {r['answer']}")
    if errs:
        print(f"\nReliable captures: {len(errs)}, mean abs error {100 * statistics.mean(errs):.0f}%, "
              f"worst {100 * max(errs):.0f}%")


def write_rows(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"Appended {len(rows)} rows to {path}")


def run_test(args):
    get, _ = settings()
    cal = dd.load_calibration(get)
    det = load_detector(args.scene_model)
    cam = cv2.VideoCapture(args.camera)
    ok, frame = cam.read()
    if not ok:
        sys.exit("cannot open the camera")
    fh, fw = frame.shape[:2]
    f_px, unc = focal(get, fw)
    print("Calibration:", cal or "none (standard door size assumed)")
    rows = []
    for ft in args.marks:
        sess = Session(cal, f_px, unc)
        k = wait_key(cam, det, sess, f"Go to the {ft:g} ft mark ({args.label}). SPACE capture, s skip, q quit")
        if k == "q":
            break
        if k == "s":
            continue
        sess = Session(cal, f_px, unc)              # fresh: no memory of the previous mark
        capture(cam, det, sess)
        now = time.time()
        raw = [m for m, _, _ in sess.raw]
        smoothed, reliable = sess.smoothed(now)
        rows.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "label": args.label, "true_ft": ft,
                     "raw_ft": round(statistics.median(raw) / FT, 2) if raw else "",
                     "smoothed_ft": round(smoothed / FT, 2) if smoothed else "",
                     "reliable": "yes" if reliable else "no",
                     "cue": sess.raw[-1][1] if sess.raw else "-",
                     "uncertainty_pct": round(100 * sess.raw[-1][2]) if sess.raw else "",
                     "calibrated": "yes" if cal else "no",
                     "first_said": sess.said[0] if sess.said else "",
                     "answer": sess.ranger.where_answer(now, 0.0, fw, f_px) or "(no door seen)"})
        print(rows[-1])
    cam.release()
    cv2.destroyAllWindows()
    if rows:
        write_rows(args.out, rows)
        report(rows)


def run_images(args):
    get, _ = settings()
    cal = dd.load_calibration(get)
    det = load_detector(args.scene_model)
    rows = []
    for path in sorted(glob.glob(os.path.join(args.images, "*"))):
        true_ft = parse_true_ft(os.path.basename(path))
        img = cv2.imread(path)
        if img is None or true_ft is None:
            continue
        fh, fw = img.shape[:2]
        f_px, unc = focal(get, fw)
        doors = doors_in(det, img, 0.0)
        m = None
        if doors:
            box = min(doors, key=lambda cb: abs((cb[1][0] + cb[1][2]) / 2 - fw / 2))[1]
            m = dd.measure(box, fw, fh, f_px, unc, cal)
        rows.append({"time": "", "label": os.path.basename(path), "true_ft": true_ft,
                     "raw_ft": round(m.metres / FT, 2) if m and m.ok else "",
                     "smoothed_ft": round(m.metres / FT, 2) if m and m.ok else "",
                     "reliable": "yes" if m and m.ok and m.unc <= dd.MAX_SYS_UNC else "no",
                     "cue": m.cue if m and m.ok else (m.reason if m else "no door"),
                     "uncertainty_pct": round(100 * m.unc) if m and m.ok else "",
                     "calibrated": "yes" if cal else "no", "first_said": "", "answer": ""})
    if not rows:
        sys.exit("no photos with a distance in the name (e.g. 6ft_facing.jpg)")
    write_rows(args.out, rows)
    report(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument("--test", action="store_true")
    mode.add_argument("--images", metavar="DIR")
    ap.add_argument("--distance-ft", type=float, default=6.0, help="calibration distance (default 6)")
    ap.add_argument("--marks", type=float, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--label", default="facing", help="what this run tests (side, dark, dim, half-hidden ...)")
    ap.add_argument("--out", default="door_distance_results.csv")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--scene-model", default="yolov8n-oiv7.pt")
    ap.add_argument("--dry-run", action="store_true", help="calibrate without saving")
    args = ap.parse_args()
    if args.calibrate:
        run_calibrate(args)
    elif args.test:
        run_test(args)
    else:
        run_images(args)


if __name__ == "__main__":
    main()
