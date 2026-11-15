"""
tools/find_cameras.py - which camera number is the Mac camera and which is Camo?

    python tools/find_cameras.py            # list, test and show every camera, pick with keys
    python tools/find_cameras.py --no-window --set camo=1 builtin=0   # save without looking

Run it once after installing Camo Studio, and again if SENSE ever opens the
wrong camera. SENSE normally finds cameras BY NAME on its own (see
camera_source.py); this tool is the "verified by a human" answer it trusts
first.

For each camera number it prints: the name macOS gives it (when known),
whether it opens, the picture size, and whether the picture is moving or
frozen (a frozen picture from Camo means the phone is not sending video).
With a window it shows each camera; press:
    B = this is the built-in Mac camera    P = this is the phone (Camo)
    N = neither / next                     Q = stop
The choice is saved in sense_memory.db (settings), SENSE uses it next start.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import camera_source as cs  # noqa: E402
import memory_db  # noqa: E402


def probe(index, seconds=1.5):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frames, t0 = [], time.time()
    while time.time() - t0 < seconds:
        ok, f = cap.read()
        if ok and f is not None and f.size:
            frames.append(f)
    cap.release()
    if not frames:
        return {"ok": False}
    small = [cv2.resize(cv2.cvtColor(cs.normalize_frame(f, None), cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
             for f in frames]
    motion = max((float(np.mean(np.abs(a - b))) for a, b in zip(small, small[1:])), default=0.0)
    return {"ok": True, "frame": frames[-1], "size": frames[-1].shape[1::-1], "fps": len(frames) / seconds,
            "frozen": motion < cs.FreezeDetector.THRESHOLD}


def save(kind, index, name):
    memory_db.set_setting(f"camera_index_{kind}", index)
    memory_db.set_setting(f"camera_name_{kind}", name)
    print(f"   saved: {kind} = camera {index} ({name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-index", type=int, default=5)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="e.g. camo=1 builtin=0")
    a = ap.parse_args()
    memory_db.init_database()

    cams = cs.list_cameras()
    names = {i: n for i, n in (cams or []) if i is not None}
    if cams is None:
        print("Camera names: not available (not macOS, or system_profiler failed).")
    else:
        print("Cameras macOS reports (OpenCV order = sorted by unique ID):")
        for i, n in cams:
            print(f"   {i if i is not None else '?'}: {n}")
        if not any(cs.matches(n, "camo") for _, n in cams):
            print("   No 'Camo' camera listed: open Camo Studio and connect the phone (USB or Wi-Fi), then run again.")

    for item in a.set:
        kind, _, idx = item.partition("=")
        if kind not in cs.LIVE_KINDS or not idx.isdigit():
            sys.exit(f"--set expects builtin=N or camo=N, got {item}")
        save(kind, int(idx), names.get(int(idx), f"camera {idx}"))
    if a.set:
        return

    found = []
    for i in range(a.max_index + 1):
        r = probe(i)
        if r is None:
            continue
        label = names.get(i, "name unknown")
        if not r["ok"]:
            print(f"Camera {i} ({label}): opens but sends no picture")
            continue
        state = "FROZEN (Camo without a phone?)" if r["frozen"] else "live"
        print(f"Camera {i} ({label}): {r['size'][0]}x{r['size'][1]}, ~{r['fps']:.0f} fps, picture {state}")
        found.append((i, label, r))
    if not found:
        print("No working camera found. Check System Settings > Privacy & Security > Camera for your terminal/VS Code.")
        return
    if a.no_window:
        print("Run without --no-window to pick, or: python tools/find_cameras.py --set camo=N builtin=M")
        return

    for i, label, r in found:
        view = cs.normalize_frame(r["frame"], 960).copy()
        cv2.putText(view, f"Camera {i}: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(view, "B = built-in Mac   P = phone (Camo)   N = next   Q = quit", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("SENSE - which camera is this?", view)
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (ord("b"), ord("B")):
                save("builtin", i, label if label != "name unknown" else f"camera {i}")
                break
            if k in (ord("p"), ord("P")):
                save("camo", i, label if label != "name unknown" else f"camera {i}")
                break
            if k in (ord("n"), ord("N")):
                break
            if k in (ord("q"), ord("Q")):
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()
    print("Done. Start SENSE with:  python sense_master.py --camera camo   (or --camera builtin)")


if __name__ == "__main__":
    main()
