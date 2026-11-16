"""
tools/ocr_eval.py - measure "read this" on the test image set, old vs new.

    python tools/make_ocr_testset.py           # once: builds ocr_testset/
    python tools/ocr_eval.py                   # all folders in ocr_testset/
    python tools/ocr_eval.py --set mine        # only your own webcam captures
    python tools/ocr_eval.py --markdown out.md # also write the table

Each image is fed to the REAL OCRController (real detector, real Tesseract) as
a still "video" (the same frame every 33 ms) after a "read this" command, and
the sentence SENSE would speak is recorded. The same is done with the old
whole-frame pipeline (use_text_detection=False) for comparison.

Per image:
  detected   a text block overlaps the labelled target (or any block, when the
             target is a whole page); for no-text images: no block on screen
  in box     share of the CHOSEN block that lies inside the labelled target
             (the label's paper is bigger than its text, so IoU would mislead)
  target     chosen block centre inside the labelled target
  recall     share of expected words that were spoken (first 40 words)
  precision  share of spoken words that are in the expected text
  narrated   SENSE produced one sentence (speech is not played here)
  honest     no "confidently wrong" answer: if recall < 0.8 (or no text is
             expected), the sentence must not be a plain "It says: ..."
"""
import argparse
import json
import os
import re
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ocr import OCRConfig, OCRController  # noqa: E402

SET_DIR = os.path.join(ROOT, "ocr_testset")


def words_of(text):
    """'200mg' and '200 mg' count the same: letters and digits are split."""
    return re.findall(r"[a-z]+|[0-9]+", text.lower())


def recall_precision(expected, spoken, max_words=40):
    """'A | B' in expected = either text is a right answer (best one counts)."""
    if " | " in expected:
        return max((recall_precision(e, spoken, max_words) for e in expected.split(" | ")),
                   key=lambda rp: rp[0] or 0)
    exp = words_of(expected)[:max_words]
    got = words_of(spoken)
    if not exp:
        return None, None
    pool = list(got)
    hit = 0
    for w in exp:
        if w in pool:
            pool.remove(w)
            hit += 1
    rec = hit / len(exp)
    pool = words_of(expected)          # extra words are judged against the WHOLE expected text
    extra_ok = 0
    for w in got:
        if w in pool:
            pool.remove(w)
            extra_ok += 1
    prec = (extra_ok / len(got)) if got else None
    return rec, prec


def read_part(sentence):
    """Only the words SENSE read out, without 'It says:' and the doubt notes."""
    for start in ("It says:", "I can only read part of it:"):
        if sentence.startswith(start):
            body = sentence[len(start):]
            for end in (". I'm not sure about", ". Some words were unclear", ". Part of the text", ". Hold it",
                        ". Move it", ". The text", ". Turn it", ". There is", ". It's too"):
                body = body.split(end)[0]
            return body
    return ""


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / u if u else 0.0


def inside(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area else 0.0


def xyxy(rect):
    x, y, w, h = rect
    return [x, y, x + w, y + h]


CONFIG_OVERRIDES = {}


def run_once(frame, objects, detection, timeout=15.0):
    logs = []
    ctrl = OCRController(config=OCRConfig(use_text_detection=detection, **CONFIG_OVERRIDES), log=logs.append)
    if ctrl.detector is not None:          # SENSE warms the detector up at start-up (OCRController.create)
        ctrl.detector.detect(frame)
    ctrl.set_objects(objects)
    t0 = time.time()
    ctrl.handle_command("read this", t0)
    msg, worst_frame = None, 0.0
    while msg is None and time.time() - t0 < timeout:
        f0 = time.perf_counter()
        now = time.time()
        ctrl.feed(frame, now)
        msg = ctrl.poll(now)
        worst_frame = max(worst_frame, time.perf_counter() - f0)
        time.sleep(0.033)
    took = time.time() - t0
    info = {"text": msg.text if msg else "(no answer)", "took": took, "worst_frame": worst_frame,
            "blocks": [xyxy(b.rect) for b in ctrl.hud_blocks],
            "chosen": xyxy(ctrl.hud_chosen.rect) if ctrl.hud_chosen is not None else None,
            "log": logs, "detector": ctrl.detector.name if ctrl.detector else "-"}
    ctrl.close()
    return info


def plain_answer(text):
    return text.startswith("It says:") and "not sure" not in text and "unclear" not in text


def score(meta, new, old):
    exp, tgt = meta["expected"], meta["target"]
    row = {"category": meta["category"]}
    blocks = new["blocks"]
    if exp:
        if tgt:
            row["detected"] = any(iou(b, tgt) > 0.1 or (tgt[0] <= (b[0] + b[2]) / 2 <= tgt[2]
                                                        and tgt[1] <= (b[1] + b[3]) / 2 <= tgt[3]) for b in blocks)
        else:
            row["detected"] = bool(blocks)
    else:
        row["detected"] = None
    ch = new["chosen"]
    row["iou"] = inside(ch, tgt) if (ch and tgt) else None
    if tgt and exp:
        row["target_ok"] = bool(ch) and tgt[0] <= (ch[0] + ch[2]) / 2 <= tgt[2] and tgt[1] <= (ch[1] + ch[3]) / 2 <= tgt[3]
    elif not exp:
        row["target_ok"] = not new["text"].startswith("It says")       # nothing chosen to narrate
    elif " | " in exp:     # several right answers: read one of them fully, or ask to frame one
        rec, _ = recall_precision(exp, read_part(new["text"]))
        row["target_ok"] = (rec or 0) >= 0.8 or "more than one place" in new["text"]
    else:
        row["target_ok"] = None
    row["recall"], row["precision"] = recall_precision(exp, read_part(new["text"]))
    row["old_recall"], row["old_precision"] = recall_precision(exp, read_part(old["text"]))
    row["narrated"] = new["text"] != "(no answer)"
    bad = (row["recall"] is None) or row["recall"] < 0.8
    row["honest"] = not (bad and plain_answer(new["text"]))
    bad_old = (row["old_recall"] is None) or row["old_recall"] < 0.8
    row["old_honest"] = not (bad_old and plain_answer(old["text"]))
    row["said"], row["old_said"] = new["text"], old["text"]
    row["took"], row["worst_frame"] = new["took"], new["worst_frame"]
    return row


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    return f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", help="subfolder(s) of ocr_testset (default: all)")
    ap.add_argument("--only", help="only files containing this text")
    ap.add_argument("--markdown", help="write the result table here")
    ap.add_argument("--json", help="write raw results here")
    ap.add_argument("--east-size", help="EAST input size WxH, e.g. 640x352")
    ap.add_argument("--rescore", help="score a --json file from an earlier run again (no OCR is run)")
    args = ap.parse_args()
    if args.east_size:
        CONFIG_OVERRIDES["east_input_size"] = tuple(int(v) for v in args.east_size.split("x"))
    sets = args.set or [d for d in ("real", "rendered", "mine") if os.path.isdir(os.path.join(SET_DIR, d))]
    rows, raw = [], {}
    if args.rescore:
        for key, v in json.load(open(args.rescore)).items():
            s, name = key.split("/", 1)
            r = score(v["meta"], v["new"], v["old"])
            r["set"], r["file"] = s, name
            rows.append(r)
        sets = sorted({r["set"] for r in rows}, key=["real", "rendered", "mine"].index)
    for s in ([] if args.rescore else sets):
        lab_path = os.path.join(SET_DIR, s, "labels.json")
        if not os.path.exists(lab_path):
            continue
        labels = json.load(open(lab_path))
        for name, meta in labels.items():
            if args.only and args.only not in name:
                continue
            frame = cv2.imread(os.path.join(SET_DIR, s, name))
            objects = [(n, tuple(b)) for n, b in meta.get("objects", [])]
            new = run_once(frame, objects, True)
            old = run_once(frame, objects, False)
            r = score(meta, new, old)
            r["set"], r["file"] = s, name
            rows.append(r)
            raw[f"{s}/{name}"] = {"new": new, "old": old, "meta": meta}
            print(f"[{s}] {name}: recall {fmt(r['recall'])} (old {fmt(r['old_recall'])}), "
                  f"target {fmt(r['target_ok'])}, honest {fmt(r['honest'])}\n   new: {r['said']}\n   old: {r['old_said']}")

    head = ("| Set | Image | Case | Text found | Chosen box inside target | Right target | Words read (new) | Words read (old) "
            "| Extra words (new) | Extra words (old) | Honest (new) | Honest (old) | Time |")
    lines = [head, "|" + " --- |" * 13]
    for r in rows:
        extra = None if r["precision"] is None else 1 - r["precision"]
        extra_old = None if r["old_precision"] is None else 1 - r["old_precision"]
        lines.append(f"| {r['set']} | {r['file']} | {r['category']} | {fmt(r['detected'])} | {fmt(r['iou'])} | "
                     f"{fmt(r['target_ok'])} | {fmt(r['recall'])} | {fmt(r['old_recall'])} | {fmt(extra)} | {fmt(extra_old)} | "
                     f"{fmt(r['honest'])} | {fmt(r['old_honest'])} | {r['took']:.1f}s |")

    def mean(key, subset):
        v = [r[key] for r in subset if r[key] is not None]
        return (sum(v) / len(v), len(v)) if v else (None, 0)

    summary = []
    for s in sets + ["all"]:
        sub = [r for r in rows if s == "all" or r["set"] == s]
        if not sub:
            continue
        parts = []
        for key, label in [("detected", "text found"), ("target_ok", "right target"), ("recall", "words read"),
                           ("old_recall", "words read (old)"), ("precision", "right share of spoken words"),
                           ("old_precision", "right share (old)"), ("honest", "honest"), ("old_honest", "honest (old)")]:
            m, n = mean(key, sub)
            parts.append(f"{label} {fmt(m)} (n={n})")
        summary.append(f"{s}: " + ", ".join(parts))
    worst = max((r["worst_frame"] for r in rows), default=0)
    summary.append(f"slowest single frame while reading: {worst * 1000:.0f} ms")
    print("\n" + "\n".join(lines) + "\n\n" + "\n".join(summary))
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write("\n".join(lines) + "\n\n" + "\n".join("- " + s for s in summary) + "\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(raw, f, indent=1, default=str)


if __name__ == "__main__":
    main()
