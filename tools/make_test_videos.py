"""
tools/make_test_videos.py - build the synthetic test clips used by tools/run_checks.py.

    python tools/make_test_videos.py OUT_DIR --objects dog.jpg --door door.jpg \
        --person-a a1.jpg a2.jpg --others b.jpg c.jpg

All clips are 1280x720 at 15 fps with walking sway, sensor noise and an
occasional motion-blurred frame. They are SYNTHETIC: a photo is zoomed and
panned to imitate walking. That tests SENSE's logic on real detector output,
but it is not a substitute for walking with the camera (see the physical
test guide).

  approach_far.mp4   walk from ~4.5 m to ~1 m towards the objects, then stand
  still_near.mp4     stand ~1.6 m from the objects (nothing should be said)
  pan_near.mp4       turn the head past the objects at ~1.6 m (nothing said)
  approach_door.mp4  walk towards a door
  find_item.mp4      see the objects, turn away, turn back, walk up ("find the ...")
  people.mp4         person A, empty room, person A (another photo), others
  sway_person.mp4    person A while the image shakes (naming while "CAM MOVING")
  events.mp4         two objects; one disappears in view, the camera turns away
                     and back, the missing one is back somewhere else
                     (--event-objects, two crops given as PHOTO:x1,y1,x2,y2)

Event clip with the darknet dog.jpg (the dog, and the bicycle without the dog):
    --event-objects dog.jpg:120,215,315,545 dog.jpg:312,115,575,425

Suggested photos: any photo with a bicycle/chair/dog in it (e.g. the classic
darknet 'dog.jpg'), any photo of a door, two different photos of one person
and photos of two other people (e.g. from the face_recognition project's
examples). No photos ship with SENSE.
"""
import argparse
import os

import cv2
import numpy as np

W, H, FPS = 1280, 720, 15


def _writer(path):
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))


def zoom_clip(img_path, out, tx, ty, z0, z1, secs, hold, pan=0.0, seed=0):
    """Fly the camera towards (tx, ty) of the photo (fractions), zoom z0 -> z1.
    The photo sits on a larger textured canvas so z < 1 (far away) works.
    pan: sweep the view sideways by this fraction of the photo width."""
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"cannot read {img_path}")
    h, w = img.shape[:2]
    s = max(W / w, H / h)
    img = cv2.resize(img, (int(w * s) + 2, int(h * s) + 2))
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    canvas = cv2.GaussianBlur(rng.normal(120, 18, (h * 3, w * 3, 3)).clip(0, 255).astype(np.uint8), (5, 5), 0)
    oy, ox = h, w
    canvas[oy:oy + h, ox:ox + w] = img
    n, m = int(secs * FPS), int(hold * FPS)
    vw = _writer(out)
    for i in range(n + m):
        z = z0 * (z1 / z0) ** (min(i, n) / max(1, n))          # constant speed in log distance
        cx = ox + (tx + pan * (i / (n + m) - 0.5)) * w + 8 * np.sin(i / 4)
        cy = oy + ty * h + 5 * np.cos(i / 6)
        x0, y0 = int(cx - W / (2 * z)), int(cy - H / (2 * z))
        f = cv2.resize(canvas[y0:y0 + int(H / z), x0:x0 + int(W / z)], (W, H))
        if i % 9 == 4:
            f = cv2.GaussianBlur(f, (9, 9), 0)
        vw.write(np.clip(f + rng.normal(0, 3, f.shape), 0, 255).astype(np.uint8))
    vw.release()
    print(out, n + m, "frames")


def find_clip(img_path, out, tx, ty, seed=0):
    """'Find the ...' test (guidance.py). The camera sees the object (4 s), turns
    LEFT away from it (3 s), waits (3 s, the question is asked here), turns back
    right (3 s), walks up to it (6 s) and stops (2 s)."""
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"cannot read {img_path}")
    h, w = img.shape[:2]
    s = max(W / w, H / h)
    img = cv2.resize(img, (int(w * s) + 2, int(h * s) + 2))
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    canvas = cv2.GaussianBlur(rng.normal(120, 18, (h * 3, w * 5, 3)).clip(0, 255).astype(np.uint8), (5, 5), 0)
    oy, ox = h, 2 * w
    canvas[oy:oy + h, ox:ox + w] = img
    turn = 1.5 * w                     # about 60 degrees at the starting zoom
    z0, z1 = 0.6, 2.6
    plan = ([(0.0, z0)] * 60 + [(-turn * k / 45, z0) for k in range(45)] + [(-turn, z0)] * 45
            + [(-turn + turn * k / 45, z0) for k in range(45)]
            + [(0.0, z0 * (z1 / z0) ** (k / 90)) for k in range(90)] + [(0.0, z1)] * 30)
    vw = _writer(out)
    for i, (dx, z) in enumerate(plan):
        cx = ox + tx * w + dx + 8 * np.sin(i / 4)
        cy = oy + ty * h + 5 * np.cos(i / 6)
        x0, y0 = int(cx - W / (2 * z)), int(cy - H / (2 * z))
        f = cv2.resize(canvas[y0:y0 + int(H / z), x0:x0 + int(W / z)], (W, H))
        if i % 9 == 4:
            f = cv2.GaussianBlur(f, (9, 9), 0)
        vw.write(np.clip(f + rng.normal(0, 3, f.shape), 0, 255).astype(np.uint8))
    vw.release()
    print(out, len(plan), "frames")


def people_clip(out, a1, a2, others):
    """A 0-59, empty 60-89, A (other photo) 90-149, then each other person 60 frames."""
    def fit(path):
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"cannot read {path}")
        h, w = img.shape[:2]
        s = min(W / w, H / h)
        img = cv2.resize(img, (int(w * s), int(h * s)))
        c = np.full((H, W, 3), 90, np.uint8)
        y, x = (H - img.shape[0]) // 2, (W - img.shape[1]) // 2
        c[y:y + img.shape[0], x:x + img.shape[1]] = img
        return c
    segs = [(a1, 60), (None, 30), (a2, 60)] + [(p, 60) for p in others]
    vw, rng, n = _writer(out), np.random.default_rng(0), 0
    for path, k in segs:
        base = fit(path) if path else np.full((H, W, 3), 90, np.uint8)
        for i in range(k):
            M = np.float32([[1, 0, 3 * np.sin(i / 5)], [0, 1, 2 * np.cos(i / 7)]])
            f = cv2.warpAffine(base, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
            vw.write(np.clip(f + rng.normal(0, 3, f.shape), 0, 255).astype(np.uint8))
        n += k
    vw.release()
    print(out, n, "frames")


def sway_person_clip(out, person, frames=240, seed=5):
    """Person A in front of the camera while the whole image shakes (about +-60 px,
    up to ~24 px per frame), so SENSE sees "CAM MOVING" and runs YOLO only on
    every 3rd frame. Regression: "this is Alex" used to fail on 2 of 3 frames."""
    rng = np.random.default_rng(seed)
    img = cv2.imread(person)
    if img is None:
        raise SystemExit(f"cannot read {person}")
    h, w = img.shape[:2]
    s = min(W / w, H / h) * 0.9
    img = cv2.resize(img, (int(w * s), int(h * s)))
    base = cv2.GaussianBlur(rng.normal(130, 25, (H + 200, W + 400, 3)).clip(0, 255).astype(np.uint8), (5, 5), 0)
    for _ in range(300):
        x, y = int(rng.integers(0, W + 340)), int(rng.integers(0, H + 150))
        cv2.rectangle(base, (x, y), (x + int(rng.integers(10, 60)), y + int(rng.integers(10, 40))),
                      tuple(int(v) for v in rng.integers(40, 230, 3)), -1)
    y0, x0 = 100 + (H - img.shape[0]) // 2, 200 + (W - img.shape[1]) // 2
    base[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    vw = _writer(out)
    for i in range(frames):
        dx = 60 * np.sin(i / 2.5)
        f = base[100:100 + H, int(200 + dx):int(200 + dx) + W]
        vw.write(np.clip(f + rng.normal(0, 3, f.shape), 0, 255).astype(np.uint8))
    vw.release()
    print(out, frames, "frames")


def _crop(spec):
    path, box = spec.rsplit(":", 1)
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    x1, y1, x2, y2 = (int(v) for v in box.split(","))
    return img[y1:y2, x1:x2].copy()


def events_clip(out, obj_a, obj_b, seed=3):
    """Event memory: A (left) and B (right) on a textured wall, slow light flicker.
       0-149   both visible
     150-314   A removed while its place is in view      -> GONE (after 8 s)
     315-359   camera turns right ~76 degrees             -> B OUT_OF_VIEW, not GONE
     360-524   looking away
     525-569   camera turns back; A is back, further right -> A moved, B just back
     570-769   both visible"""
    rng = np.random.default_rng(seed)
    a, b = _crop(obj_a), _crop(obj_b)
    cw = 4800
    canvas = cv2.GaussianBlur(rng.normal(150, 20, (H, cw, 3)).clip(0, 255).astype(np.uint8), (5, 5), 0)
    for _ in range(400):   # wall texture: corners for the optical-flow heading
        x, y = int(rng.integers(0, cw - 60)), int(rng.integers(0, int(H * 0.5)))
        cv2.rectangle(canvas, (x, y), (x + int(rng.integers(10, 60)), y + int(rng.integers(10, 40))),
                      tuple(int(v) for v in rng.integers(40, 230, 3)), -1)
    canvas[int(H * 0.75):] = (95, 105, 120)
    view0 = 1500

    def paste(img, obj, x, y):
        img[y:y + obj.shape[0], x:x + obj.shape[1]] = obj

    plan = ([(view0, 150)] * 150 + [(view0, None)] * 165
            + [(view0 + 1500 * k / 45, None) for k in range(45)] + [(view0 + 1500, None)] * 165
            + [(view0 + 1500 - 1500 * k / 45, 1000) for k in range(45)] + [(view0, 1000)] * 200)
    vw = _writer(out)
    for i, (vx, ax) in enumerate(plan):
        img = canvas.copy()
        paste(img, b, view0 + 650, 300)
        if ax is not None:
            paste(img, a, view0 + ax, 330)
        f = img[:, int(vx):int(vx) + W].astype(np.float32) * (1.0 + 0.12 * np.sin(i / 20))
        vw.write(np.clip(f + rng.normal(0, 3, f.shape), 0, 255).astype(np.uint8))
    vw.release()
    print(out, len(plan), "frames")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--objects", help="photo with objects to walk towards (target: its centre)")
    ap.add_argument("--door", help="photo of a door")
    ap.add_argument("--person-a", nargs=2, metavar=("PHOTO1", "PHOTO2"))
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--event-objects", nargs=2, metavar=("A", "B"),
                    help="two object crops as PHOTO:x1,y1,x2,y2 for events.mp4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    j = lambda name: os.path.join(a.out, name)
    if a.objects:
        zoom_clip(a.objects, j("approach_far.mp4"), 0.45, 0.5, 0.5, 2.2, 6, 3)
        zoom_clip(a.objects, j("still_near.mp4"), 0.45, 0.5, 1.4, 1.4, 0.1, 8)
        zoom_clip(a.objects, j("pan_near.mp4"), 0.45, 0.5, 1.3, 1.3, 0.1, 8, pan=0.8)
        find_clip(a.objects, j("find_item.mp4"), 0.45, 0.5)
    if a.door:
        zoom_clip(a.door, j("approach_door.mp4"), 0.5, 0.5, 0.6, 3.0, 7, 3)
    if a.person_a:
        people_clip(j("people.mp4"), a.person_a[0], a.person_a[1], a.others)
        sway_person_clip(j("sway_person.mp4"), a.person_a[0])
    if a.event_objects:
        events_clip(j("events.mp4"), *a.event_objects)


if __name__ == "__main__":
    main()
