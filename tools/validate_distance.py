"""
tools/validate_distance.py - measure how wrong SENSE's distances are, and calibrate.

Do not trust a distance formula you have not checked against a tape measure.
This script compares SENSE's estimate with the TRUE distance you measured.

LIVE (webcam):
    python tools/validate_distance.py --live --distances 0.5 1 1.5 2 3
    Stand at each tape-measured distance (camera lens to your face/chest),
    press SPACE to capture, q to stop early.

FROM PHOTOS (repeatable): name each photo after its true distance, e.g.
    calib/100cm.jpg, calib/150cm_side.jpg, calib/2.5m.jpg
    python tools/validate_distance.py --images calib

It prints, per capture: true distance, which cue was used (eyes / height /
width / upper-bound), the estimate with its range, and the error.

CALIBRATION: if you also give your real height (--person-height 1.68) and
stand far enough that your whole body is visible (about 3 m or more), or your
face is visible (eye cue, once face recognition is installed), the script
computes the focal length that best fits YOUR camera. Add --save to store it
in sense_memory.db; sense_master.py then uses it on the next start.
"""
import argparse
import glob
import os
import re
import statistics
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import distance  # noqa: E402

try:                         # optional: eye-distance cue (audit fix 4)
    import face_id
except Exception:            # noqa: BLE001
    face_id = None


def true_distance_from_name(path):
    name = os.path.basename(path).lower()
    m = re.match(r"(\d+(?:\.\d+)?)\s*(cm|m)", name)
    if not m:
        return None
    value = float(m.group(1))
    return value / 100.0 if m.group(2) == "cm" else value


def largest_person(model, frame):
    res = model(frame, classes=[0], conf=0.4, verbose=False)[0]
    boxes = [tuple(float(v) for v in b.xyxy[0]) for b in res.boxes]
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if boxes else None


def measure(model, faces, frame, focal_ratio, person_height):
    h, w = frame.shape[:2]
    box = largest_person(model, frame)
    if box is None:
        return None
    box = (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))
    f_px = distance.focal_px(w, focal_ratio=focal_ratio)
    eye_px = faces.eye_distance_px(frame, box) if faces else None
    est = distance.person_distance(box, w, h, f_px, eye_px=eye_px)
    return {"box": box, "est": est, "f_px": f_px, "w": w, "h": h, "eye_px": eye_px,
            "person_height": person_height}


def implied_ratio(m, truth):
    """Focal length (as f / width) that would make this capture exactly right."""
    est, box = m["est"], m["box"]
    if est.cue == "eyes":
        return truth * m["eye_px"] / distance.EYE_DISTANCE[0] / m["w"]
    if est.cue == "height" and m["person_height"]:
        return truth * (box[3] - box[1]) / m["person_height"] / m["w"]
    return None


def report(rows, focal_ratio, save):
    print(f"\n{'true':>6} {'cue':>12} {'estimate':>9} {'range':>13} {'error':>7}  in range?  spoken")
    errors, ratios = [], []
    for truth, m in rows:
        est = m["est"]
        spoken = distance.phrase(est)
        if est.metres is not None:
            err = (est.metres - truth) / truth
            errors.append(abs(err))
            inside = est.low <= truth <= est.high
            print(f"{truth:5.2f}m {est.cue:>12} {est.metres:8.2f}m {est.low:5.2f}-{est.high:5.2f}m "
                  f"{err:+7.0%}  {'yes' if inside else 'NO':>9}  {spoken}")
        else:
            inside = est.high is not None and truth <= est.high
            hi = f"< {est.high:.2f}m" if est.high else "unknown"
            print(f"{truth:5.2f}m {est.cue:>12} {'-':>9} {hi:>13} {'-':>7}  {'yes' if inside else 'NO':>9}  {spoken}")
        r = implied_ratio(m, truth)
        if r:
            ratios.append(r)
    if errors:
        print(f"\nMedian absolute error: {statistics.median(errors):.0%} over {len(errors)} captures "
              f"({'calibrated' if focal_ratio else 'NOT calibrated'}).")
    if ratios:
        best = statistics.median(ratios)
        print(f"Best-fit focal length: {best:.3f} x frame width "
              f"(assumed 65 degree view gives {distance.focal_px(1) :.3f}).")
        if save:
            import memory_db
            memory_db.init_database()
            memory_db.set_setting("focal_ratio", f"{best:.4f}")
            print("Saved. sense_master.py will use it on the next start.")
    elif save:
        print("Nothing to calibrate from: give --person-height and stand far enough to be fully visible, "
              "or install face recognition for the eye cue.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="folder of photos named by true distance, e.g. 150cm.jpg")
    ap.add_argument("--live", action="store_true", help="use the webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--distances", type=float, nargs="*", default=[0.5, 1, 1.5, 2, 3])
    ap.add_argument("--person-height", type=float, help="your real height in metres, for calibration")
    ap.add_argument("--save", action="store_true", help="store the best-fit focal length")
    ap.add_argument("--uncalibrated", action="store_true", help="ignore a stored calibration")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(os.path.join(ROOT, "yolov8s.pt"))
    faces = face_id.FaceID.load_if_available() if face_id else None

    focal_ratio = None
    if not args.uncalibrated:
        import memory_db
        try:
            focal_ratio = float(memory_db.get_setting("focal_ratio", "0")) or None
        except ValueError:
            focal_ratio = None

    rows = []
    if args.images:
        for path in sorted(glob.glob(os.path.join(args.images, "*"))):
            truth = true_distance_from_name(path)
            frame = cv2.imread(path)
            if truth is None or frame is None:
                continue
            m = measure(model, faces, frame, focal_ratio, args.person_height)
            if m is None:
                print(f"{os.path.basename(path)}: no person found")
                continue
            rows.append((truth, m))
    elif args.live:
        cam = cv2.VideoCapture(args.camera)
        for truth in args.distances:
            print(f"Stand at {truth} m and press SPACE (q to stop).")
            while True:
                ok, frame = cam.read()
                if not ok:
                    break
                cv2.putText(frame, f"Stand at {truth} m, press SPACE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("validate distance", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    ok, frame = cam.read()
                    m = measure(model, faces, frame, focal_ratio, args.person_height)
                    if m is None:
                        print("No person found, try again.")
                        continue
                    rows.append((truth, m))
                    break
                if key == ord("q"):
                    break
            if key == ord("q"):
                break
        cam.release()
        cv2.destroyAllWindows()
    else:
        ap.error("give --images FOLDER or --live")
    if rows:
        report(rows, focal_ratio, args.save)
    else:
        print("No captures.")


if __name__ == "__main__":
    main()
