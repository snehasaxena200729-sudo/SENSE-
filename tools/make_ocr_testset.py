"""
tools/make_ocr_testset.py - build the OCR test image set used by tools/ocr_eval.py.

    python tools/make_ocr_testset.py            # -> ocr_testset/ (git-ignored)

Two kinds of images, kept apart in the results so nobody mistakes one for the other:

  real/      public photos with text, downloaded once (needs internet once):
             signs, a poster, book pages, a document, and two frames from the
             ICDAR 2015 "incidental scene text" set, which was filmed with a
             head-mounted camera (close to how SENSE sees). Expected texts were
             typed by hand from the images.
  rendered/  printed-style labels (medicine bottle, medicine packet, room label,
             sign, document...) pasted into real room photos and degraded the
             way a webcam degrades them: small, tilted, leaning back, blurry,
             dark, noisy, JPEG. Exact ground truth, but NOT real camera captures.

Your own webcam captures (tools/ocr_capture.py) go in mine/ with the same
labels.json format and are scored the same way.

labels.json: {file: {"category", "expected", "target": [x1, y1, x2, y2] or null,
                     "objects": [[name, [x1, y1, x2, y2]], ...], "note"}}
"expected" is the text SENSE should read for the target ("" = no readable text).

Rendering needs Pillow (pip install pillow) for real fonts; it is a test-only
tool, SENSE itself does not use Pillow.
"""
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "ocr_testset")
W, H = 1280, 720

REAL = {
    # file: (url, category, expected, target box in the ORIGINAL image or None, crop or None)
    "who_poster.png": ("https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/english.png",
                       "document", "Reduce your risk of coronavirus infection: Clean hands with soap and water "
                       "or alcohol-based hand rub Cover nose and mouth when coughing and sneezing with tissue "
                       "or flexed elbow Avoid close contact with anyone with cold or flu-like symptoms "
                       "Thoroughly cook meat and eggs No unprotected contact with live wild or farm animals "
                       "World Health Organization", None, None),
    # two separate signs: reading either one fully, or asking to frame one, is right
    "street_signs.png": ("https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/example3.png",
                         "multiple regions", "BASRURKAR MARKET | SPEED LIMIT", None, (0, 0, 231, 150)),
    "book_page.png": ("https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/imageTextN.png",
                      "book", "technical details are too complex to cover in the book itself. In teaching our "
                      "courses, we have found it useful for the students to attempt a number of small "
                      "implementation projects, which often build on one another, in order to get them used to "
                      "working with real-world images and the challenges that these present.", None, None),
    "book_page_tilted.png": ("https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/imageTextR.png",
                             "tilted", "technical details are too complex to cover in the book itself. In "
                             "teaching our courses, we have found it useful for the students to attempt a "
                             "number of small implementation projects, which often build on one another, in "
                             "order to get them used to working with real-world images and the challenges "
                             "that these present.", None, None),
    "video_sign.png": ("https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/dnn/text_det_test1.png",
                       "signboard", "Video", [160, 140, 300, 230], None),
    "eurotext.png": ("https://raw.githubusercontent.com/tesseract-ocr/tessdoc/main/images/eurotext.png",
                     "document", "The (quick) [brown] {fox} jumps! Over the $43,456.78 <lazy> #90 dog "
                     "& duck/goose, as 12.5% of E-mail from aspammer@website.com is spam.", None, None),
    "icdar_room_sign.jpg": ("https://raw.githubusercontent.com/argman/EAST/master/training_samples/img_1.jpg",
                            "room label", "Genaxis Theatre [06] 62-03 Carpark", [340, 0, 760, 220], None),
    "icdar_far_exit.jpg": ("https://raw.githubusercontent.com/argman/EAST/master/training_samples/img_2.jpg",
                           "small text", "EXIT", [598, 168, 640, 200], None),
}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf", "/Library/Fonts/Arial.ttf",
]


def fonts():
    found = [f for f in FONT_CANDIDATES if os.path.exists(f)]
    return found or [None]


def fetch(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)


def text_card(lines, size, bg=(240, 240, 235), fg=(20, 20, 20), font_i=0, pad=0.6):
    """A label image (BGR) with the given lines, letter height ~ size px."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        fpath = fonts()[font_i % len(fonts())]
        font = ImageFont.truetype(fpath, int(size * 1.35)) if fpath else ImageFont.load_default()
        tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        boxes = [tmp.textbbox((0, 0), ln, font=font) for ln in lines]
        lw = max(b[2] - b[0] for b in boxes)
        lh = int(size * 1.9)
        p = int(size * pad * 2)
        img = Image.new("RGB", (lw + 2 * p, lh * len(lines) + 2 * p), bg[::-1])
        d = ImageDraw.Draw(img)
        for i, ln in enumerate(lines):
            d.text((p, p + i * lh), ln, fill=fg[::-1], font=font)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except ImportError:
        scale = size / 22.0
        lh = int(size * 1.9)
        lw = max(cv2.getTextSize(ln, cv2.FONT_HERSHEY_DUPLEX, scale, 2)[0][0] for ln in lines)
        p = int(size * pad * 2)
        img = np.full((lh * len(lines) + 2 * p, lw + 2 * p, 3), bg, np.uint8)
        for i, ln in enumerate(lines):
            cv2.putText(img, ln, (p, p + i * lh + size), cv2.FONT_HERSHEY_DUPLEX, scale, fg, 2, cv2.LINE_AA)
        return img


def cylinder(card, strength=0.35):
    """Wrap a flat label around a bottle: the sides get squeezed (curved text)."""
    h, w = card.shape[:2]
    xs = np.linspace(-1, 1, w)
    t = strength * np.pi                      # half the visible arc of the bottle
    # a point at angle a on the label appears at x = sin(a)/sin(t) on screen
    src_x = (np.arcsin(xs * np.sin(t)) / t + 1) / 2 * (w - 1)
    mapx = np.tile(src_x.astype(np.float32), (h, 1))
    mapy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    out = cv2.remap(card, mapx, mapy, cv2.INTER_LINEAR)
    shade = (0.55 + 0.45 * np.cos(xs * np.pi / 2.2))[None, :, None]
    return np.clip(out * shade, 0, 255).astype(np.uint8)


def paste(bg, card, centre, angle=0.0, lean=0.0, scale=1.0):
    """Paste card into bg at centre, rotated by angle (deg), leaning back by `lean`
    (0..0.5, top edge shorter). Returns (image, target box x1 y1 x2 y2)."""
    h, w = card.shape[:2]
    w2, h2 = w * scale, h * scale
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dx = w2 * lean / 2
    dst = np.float32([[-w2 / 2 + dx, -h2 / 2], [w2 / 2 - dx, -h2 / 2], [w2 / 2, h2 / 2], [-w2 / 2, h2 / 2]])
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    dst = dst @ R.T + np.float32(centre)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(card, M, (bg.shape[1], bg.shape[0]))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (bg.shape[1], bg.shape[0]))
    out = bg.copy()
    out[mask > 0] = warped[mask > 0]
    x1, y1 = dst.min(axis=0)
    x2, y2 = dst.max(axis=0)
    return out, [int(max(0, x1)), int(max(0, y1)), int(min(bg.shape[1], x2)), int(min(bg.shape[0], y2))]


def webcam(img, blur=0, motion=0, dark=1.0, noise=4.0, seed=0, jpeg=80):
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) * dark
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if motion:
        k = np.zeros((motion, motion), np.float32)
        k[motion // 2, :] = 1.0 / motion
        out = cv2.filter2D(out, -1, k)
    out = np.clip(out + rng.normal(0, noise * (1.6 if dark < 0.6 else 1.0), out.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def backgrounds(cache):
    urls = {"home": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/home.jpg",
            "building": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/building.jpg",
            "bus": "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg",
            "zidane": "https://raw.githubusercontent.com/ultralytics/assets/main/im/zidane.jpg"}
    out = {}
    for k, u in urls.items():
        p = os.path.join(cache, k + ".jpg")
        fetch(u, p)
        im = cv2.imread(p)
        out[k] = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    return out


def bottle(bg, label_lines, size, centre, curve=0.35, seed=0):
    """A white bottle body with a curved label, held in front of the camera."""
    card = text_card(label_lines, size, bg=(250, 250, 250), fg=(30, 30, 140), font_i=0)
    card = cylinder(card, curve)
    h, w = card.shape[:2]
    body = np.full((int(h * 2.2), int(w * 1.04), 3), (215, 215, 210), np.uint8)
    body = cylinder(body, curve)
    by, bx = (body.shape[0] - h) // 2, (body.shape[1] - w) // 2
    body[by:by + h, bx:bx + w] = card
    img, box = paste(bg, body, centre)
    cy = centre[1]
    tgt = [box[0] + bx, int(cy - h / 2), box[0] + bx + w, int(cy + h / 2)]
    return img, tgt, box


def rendered(cache):
    bgs = backgrounds(cache)
    cases = {}

    def add(name, img, category, expected, target, objects=None, note=""):
        cases[name] = (img, {"category": category, "expected": expected, "target": target,
                             "objects": objects or [], "note": note})

    c = (W // 2, H // 2 + 20)
    # medicine bottles
    lines = ["PARACETAMOL", "500 mg Tablets", "Take 1 every 6 hours"]
    img, t, body = bottle(bgs["home"], lines, 26, c)
    add("med_bottle.jpg", webcam(img, seed=1), "medicine bottle", " ".join(lines), t, [["bottle", body]],
        "curved label, held at about 30 cm")
    img, t, body = bottle(bgs["building"], ["IBUPROFEN 200 mg", "Do not exceed 6 in 24 hours"], 18, c, seed=2)
    add("med_bottle_small.jpg", webcam(img, seed=2), "medicine bottle",
        "IBUPROFEN 200 mg Do not exceed 6 in 24 hours", t, [["bottle", body]], "smaller print")
    # medicine packets (flat box front)
    lines = ["AMOXICILLIN", "250 mg Capsules", "21 capsules"]
    card = text_card(lines, 30, bg=(245, 245, 250), fg=(120, 40, 20), font_i=2)
    img, t = paste(bgs["home"], card, c)
    add("med_packet.jpg", webcam(img, seed=3), "medicine packet", " ".join(lines), t)
    lines = ["CETIRIZINE 10 mg", "Allergy relief", "Take one tablet daily"]
    card = text_card(lines, 24, bg=(230, 245, 230), fg=(20, 70, 20), font_i=1)
    img, t = paste(bgs["building"], card, (c[0] + 40, c[1]), angle=8, lean=0.15)
    add("med_packet_angle.jpg", webcam(img, seed=4), "medicine packet", " ".join(lines), t,
        note="slightly rotated and leaning back")
    # printed document and book page
    lines = ["Appointment letter", "Date: 18 November 2026", "Time: 10:30 am", "Room 12, Ground Floor",
             "Please bring this letter"]
    card = text_card(lines, 20, bg=(250, 250, 250), fg=(10, 10, 10), font_i=4, pad=1.2)
    img, t = paste(bgs["home"], card, c, scale=1.0)
    add("document.jpg", webcam(img, seed=5), "document", " ".join(lines), t)
    lines = ["Chapter 3", "The morning was cold and", "quiet when she left the", "house for the station."]
    card = text_card(lines, 22, bg=(225, 235, 240), fg=(25, 25, 25), font_i=4, pad=1.0)
    img, t = paste(bgs["zidane"], card, c, angle=-3)
    add("book.jpg", webcam(img, seed=6), "book", " ".join(lines), t, note="person in background")
    # room label and sign boards
    card = text_card(["ROOM 204"], 46, bg=(40, 40, 40), fg=(245, 245, 245), font_i=0)
    img, t = paste(bgs["building"], card, (c[0], c[1] - 120))
    add("room_label.jpg", webcam(img, seed=7), "room label", "ROOM 204", t, note="light text on dark plate")
    card = text_card(["WET FLOOR", "CAUTION"], 40, bg=(20, 200, 240), fg=(10, 10, 10), font_i=0)
    img, t = paste(bgs["bus"], card, c, lean=0.2)
    add("signboard.jpg", webcam(img, seed=8), "signboard", "WET FLOOR CAUTION", t, note="leaning back")
    # small text
    lines = ["Batch 2611A", "Exp 03/2028"]
    card = text_card(lines, 10, bg=(245, 245, 245), fg=(20, 20, 20), font_i=0)
    img, t = paste(bgs["home"], card, c)
    add("small_text.jpg", webcam(img, seed=9), "small text", " ".join(lines), t, note="letters ~10 px")
    # tilted
    lines = ["VITAMIN D3", "1000 IU"]
    card = text_card(lines, 30, bg=(240, 240, 240), fg=(140, 20, 20), font_i=0)
    img, t = paste(bgs["home"], card, c, angle=22)
    add("tilted_22.jpg", webcam(img, seed=10), "tilted", " ".join(lines), t, note="rotated 22 degrees")
    img, t = paste(bgs["building"], card, c, angle=-35)
    add("tilted_35.jpg", webcam(img, seed=11), "tilted", " ".join(lines), t, note="rotated 35 degrees")
    # blurry
    lines = ["ASPIRIN 75 mg", "Gastro-resistant"]
    card = text_card(lines, 28, bg=(245, 245, 245), fg=(20, 20, 120), font_i=0)
    img, t = paste(bgs["home"], card, c)
    add("blurry_soft.jpg", webcam(img, blur=1.6, seed=12), "blurry", " ".join(lines), t, note="slight defocus")
    add("blurry_motion.jpg", webcam(img, motion=15, seed=13), "blurry", " ".join(lines), t,
        note="hand shake (15 px motion blur)")
    add("blurry_heavy.jpg", webcam(img, blur=4.5, seed=14), "blurry", " ".join(lines), t,
        note="heavy blur: should NOT be narrated, should ask to hold still")
    add("dim_light.jpg", webcam(img, dark=0.28, noise=6, seed=15), "dim light", " ".join(lines), t,
        note="dark room, grainy")
    # several text regions in one frame
    bg = bgs["home"].copy()
    poster = text_card(["SUMMER SALE", "50% OFF"], 18, bg=(30, 30, 180), fg=(255, 255, 255), font_i=0)
    bg, _ = paste(bg, poster, (190, 130))
    lines = ["METFORMIN 500 mg", "Take with food"]
    card = text_card(lines, 30, bg=(245, 245, 245), fg=(20, 20, 20), font_i=0)
    img, t = paste(bg, card, (c[0] + 30, c[1] + 40))
    add("multi_held_and_poster.jpg", webcam(img, seed=16), "multiple regions", " ".join(lines), t,
        note="small poster top-left, packet held in the middle")
    bg = bgs["building"].copy()
    a = text_card(["EXIT"], 30, bg=(20, 140, 20), fg=(255, 255, 255))
    b = text_card(["TOILETS"], 30, bg=(200, 200, 200), fg=(20, 20, 20))
    bg, _ = paste(bg, a, (360, 360))
    img, _ = paste(bg, b, (920, 360))
    add("multi_two_equal.jpg", webcam(img, seed=17), "multiple regions", "", None,
        note="two equal signs left and right: SENSE must NOT pick one silently")
    # no readable text
    add("no_text_room.jpg", webcam(bgs["home"], seed=18), "no text", "", None, note="room, no text")
    body = np.full((260, 150, 3), (200, 200, 205), np.uint8)
    body = cylinder(body, 0.35)
    img, box = paste(bgs["building"], body, c)
    add("no_text_bottle.jpg", webcam(img, seed=19), "no text", "", None, [["bottle", box]],
        note="bottle with the label turned away")
    return cases


def main():
    os.makedirs(os.path.join(OUT, "real"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "rendered"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "mine"), exist_ok=True)
    cache = os.path.join(OUT, "cache")
    os.makedirs(cache, exist_ok=True)
    labels = {}
    for name, (url, cat, expected, target, crop) in REAL.items():
        raw = os.path.join(cache, name)
        try:
            fetch(url, raw)
        except Exception as e:
            print(f"skip {name}: {e}")
            continue
        im = cv2.imread(raw)
        if crop:
            x0, y0, x1, y1 = crop
            im = im[y0:y1, x0:x1]
        cv2.imwrite(os.path.join(OUT, "real", name), im)
        labels[name] = {"category": cat, "expected": expected, "target": target, "objects": [],
                        "note": "public photo"}
    with open(os.path.join(OUT, "real", "labels.json"), "w") as f:
        json.dump(labels, f, indent=1)
    labels = {}
    for name, (img, meta) in rendered(cache).items():
        cv2.imwrite(os.path.join(OUT, "rendered", name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        labels[name] = meta
    with open(os.path.join(OUT, "rendered", "labels.json"), "w") as f:
        json.dump(labels, f, indent=1)
    print(f"Test set written to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
