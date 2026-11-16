"""
tools/stress_interactive.py - 10,000-case stress test of the interactive/ layer.

    python tools/stress_interactive.py [--seed 7] [--json out.json]

Seeded, so a failure can be reproduced exactly. No camera, YOLO, mic or
Ollama: each case builds a random scene with KNOWN truth (objects, colours,
sides, people, memory) and checks the spoken answer against that truth.

  A  5,000 new-style questions (colour, count, sides, safety, describe,
     earlier/behind, date, time, reading) with Whisper-style noise
  B  3,000 existing-command phrasings: Path B must return None and say
     nothing, so existing SENSE behaviour is unchanged
  C  1,000 follow-up chains ("what colour is the cup?" -> "where is it?")
  D  1,000 local-AI-model answers (grounded and hallucinated) through the
     fact-check in interactive/guard.py

Pass rules: zero failed checks, zero errors inside the layer, every answer
under 20 ms.
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from interactive import InteractiveQA, facts as F, guard as G, text as T  # noqa: E402
from interactive_fakes import BGR, FakeMemory, crop_of, make_hooks  # noqa: E402

FW, FH = 1280, 720
OBJ = ["bottle", "cup", "phone", "book", "laptop", "chair", "remote", "mouse", "couch", "tv", "plant", "bed"]
KNOWN, LEGACY = ["Sneha", "Krish"], ["Rahul"]
SIDE_X = {"left": (110, 360), "ahead": (500, 780), "right": (920, 1170)}
SIDE_WORDS = {"left": "on your left", "right": "on your right", "ahead": "in front of you"}
MAX_MS = 20.0


class Failures:
    def __init__(self):
        self.items, self.count = [], 0

    def add(self, part, case, q, answer, why):
        self.count += 1
        if len(self.items) < 40:
            self.items.append({"part": part, "case": case, "question": q, "answer": answer, "why": why})


def noisy(rng, text, heavy=True):
    t = text
    if rng.random() < 0.3:
        t = t.capitalize()
    if rng.random() < 0.4:
        t += rng.choice(["?", ".", "", " please", "?"])
    if heavy and rng.random() < 0.25:
        t = rng.choice(["um ", "okay ", "hey sense, ", "sense ", "so "]) + t
    if rng.random() < 0.3:
        t = t.replace("colour", "color")
    if heavy and rng.random() < 0.08:
        t = t.replace("bottle", "bottel")
    return t


def plural(c):
    return T.plural(c, 2)


# ------------------------------------------------------------------ scenes
def make_scene(rng, now):
    mem = FakeMemory()
    h = make_hooks(memory=mem, known=KNOWN, legacy=LEGACY, heading=rng.choice([0.0, 0.0, 90.0, -90.0, 180.0]))
    qa = InteractiveQA(h, log=lambda *_: None)
    dark = rng.random() < 0.08
    items = []
    for tid in range(1, rng.randint(0, 7) + 1):
        person = rng.random() < 0.22
        cls = "person" if person else rng.choice(OBJ)
        side = rng.choice(list(SIDE_X))
        colour = rng.choice(list(BGR))
        w, hgt = rng.randint(60, 160), rng.randint(80, 200)
        cx = rng.randint(*SIDE_X[side])
        x1, y1 = max(1 + w // 2, cx) - w // 2, rng.randint(150, 450)
        kind = "normal"
        r = rng.random()
        if r < 0.08 and side == "left":
            x1, kind = 0, "partial"
        elif r < 0.13:
            w, hgt, kind = 20, 20, "small"
        box = (x1, y1, x1 + w, y1 + hgt)
        crop = crop_of(colour, h=hgt, w=w, noise=rng.choice([0, 6, 12]), rng=np.random.default_rng(rng.randint(0, 10**6)),
                       dark=dark)
        name = None
        if person:
            name = rng.choice(KNOWN) if rng.random() < 0.4 else None
            label = name or "Person"
        elif rng.random() < 0.1:
            label = f"My {cls.title()}"
        else:
            voted = colour if rng.random() < 0.8 else rng.choice(list(BGR))
            label = f"{voted.title()} {cls.title()}"
        dist = rng.choice(["about 2 to 3 feet away", "within arm's reach", "distance unclear", "far away"])
        qa.see(tid, cls, label, name, box, crop, dist, None, FW, FH)
        items.append({"tid": tid, "cls": cls, "side": side, "colour": colour, "kind": kind, "person": name,
                      "label": label})
    gray = np.full((90, 160), 20 if dark else 140, np.uint8)
    stale = rng.random() < 0.05
    qa.frame_done(now - (5.0 if stale else 0.1), FW, FH, gray)
    # memory: some things seen earlier, at known world angles
    for _ in range(rng.randint(0, 6)):
        c = rng.choice(OBJ)
        col = rng.choice(list(BGR))
        mem.add(f"{col} {c}", now - rng.randint(5, 1700), world_angle=float(rng.randint(-179, 179)),
                session_id=rng.choice(["S", "S", "S", "OLD"]))
    if rng.random() < 0.3:
        mem.add("rahul", now - 50, world_angle=0.0)      # shirt-colour sighting: must stay hidden
    return qa, h, mem, items, {"dark": dark, "stale": stale}


def visible_people_names(items):
    return {i["person"] for i in items if i["person"]}


def timed(qa, q, now, lat):
    t0 = time.perf_counter()
    r = qa.handle(q, now)
    lat.append((time.perf_counter() - t0) * 1000)
    return r


# ------------------------------------------------------------------ part A
def part_a(rng, n, fails, lat):
    by_type = {}
    for case in range(n):
        now = 1_763_300_000.0 + case * 37
        qa, h, mem, items, sc = make_scene(rng, now)
        kind = rng.choice(["COLOUR", "COLOUR", "COUNT", "COUNT", "SIDE", "SIDE", "SAFETY", "DESCRIBE",
                           "PAST", "PAST", "TIME", "DATE", "READ"])
        cls = rng.choice(OBJ + ["person"] * 2)
        side = rng.choice(["left", "right", "ahead"])
        if kind == "COLOUR":
            cls = rng.choice(OBJ)
            q = rng.choice([f"what colour is the {cls}", f"tell me the colour of the {cls}",
                            f"what colour is the {cls} in front of me" if False else f"which colour is the {cls}"])
        elif kind == "COUNT":
            q = rng.choice([f"how many {plural(cls)} are there", f"how many {plural(cls)} can you see",
                            f"how many {plural(cls)} are in front of me", f"how many {plural(cls)} are on my {side}"
                            if side != "ahead" else f"count the {plural(cls)}"])
            if cls == "person":
                q = rng.choice(["how many people are around me", "how many people can you see",
                                f"how many people are on my {side}" if side != "ahead" else "how many people are there"])
        elif kind == "SIDE":
            sw = {"left": "on my left", "right": "on my right", "ahead": "in front of me"}[side]
            q = rng.choice([f"is there anything {sw}", f"what is {sw}", f"anything {sw}", f"is there a {cls} {sw}"
                            if cls != "person" else f"is there anyone {sw}"])
        elif kind == "SAFETY":
            side = "ahead"
            q = rng.choice(["is it safe to walk forward", "can i walk ahead", "is the path clear", "any obstacles ahead",
                            "is anything in my way"])
        elif kind == "DESCRIBE":
            q = rng.choice([f"describe the {cls}" if cls != "person" else "describe the person",
                            "describe what is in front of me", "what am i holding", f"tell me about the {cls}"])
        elif kind == "PAST":
            side = rng.choice(["left", "right", "ahead", "behind"])
            q = {"left": "what was on my left before", "right": "what was on my right earlier",
                 "ahead": "what was previously in front of me", "behind": "what is behind me"}[side]
        else:
            q = {"TIME": "what time is it", "DATE": "what is today's date", "READ": "what does this label say"}[kind]
        q = noisy(rng, q)
        r = timed(qa, q, now, lat)
        by_type[kind] = by_type.get(kind, 0) + 1
        a = r.text if r else None
        tag = (case, q, a)

        def fail(why):
            fails.add("A", *tag, why)

        if qa.stats["errors"]:
            fail("error inside layer")
            continue
        if r is None:
            if not (kind == "DESCRIBE" and "tell me about" in q.lower() and False):
                fail("declined a Path B question")
            continue
        if r.kind == "rewrite":
            if kind != "DESCRIBE":
                fail("unexpected rewrite")
            continue
        low = a.lower()
        if "rahul" in low:
            fail("named an unverified (clothing-colour) person")
        for nm in KNOWN:
            if nm.lower() in low and nm not in visible_people_names(items) and kind not in ("PAST",):
                fail(f"named {nm} who is not visible")
        if sc["stale"] and kind in ("COLOUR", "COUNT", "SIDE", "DESCRIBE") and \
                not re.search(r"clear look|can't see clearly|can't see the|i don't see|remember", low):
            if kind != "COUNT" or cls not in ("door", "stairs"):
                fail("answered from a stale frame as if live")
        if kind == "TIME" and not re.match(r"It is \d{1,2}:\d\d (AM|PM)\.$", a):
            fail("bad time")
        if kind == "DATE" and not a.startswith("Today is "):
            fail("bad date")
        if kind == "READ" and a != "Reading. Hold it steady.":
            fail("did not hand reading to OCR")
        if kind == "SAFETY" and not re.search(r"can't promise|can't detect everything|can't check", low):
            fail("safety answer without caution")
        if kind == "SAFETY" and re.search(r"\b(it is|it.s) safe\b|^(the )?(path|way) is clear", low):
            fail("promised safety")
        if sc["stale"]:
            continue
        if kind == "COUNT":
            if "how many people" in q.lower() or cls == "person":
                pool = [i for i in items if i["cls"] == "person"]
                target = "person"
            else:
                pool = [i for i in items if i["cls"] == cls]
                target = cls
            m = re.search(r"on my (left|right)", q.lower())
            if m:
                pool = [i for i in pool if i["side"] == m.group(1)]
            n_true = len(pool)
            if n_true == 0 and "don't see" not in low:
                fail(f"count: expected none, truth 0 {target}")
            if n_true > 0 and f"{T.number_word(n_true)} " not in low + " ":
                fail(f"count: expected {n_true} {target}")
        if kind == "COLOUR":
            vis = [i for i in items if i["cls"] == cls]
            if len(vis) == 1:
                i = vis[0]
                if sc["dark"]:
                    if "too dark" not in low and "too small" not in low:
                        fail("dark scene not reported")
                elif i["kind"] == "small":
                    if "too small" not in low:
                        fail("small crop not reported")
                elif i["kind"] == "partial":
                    if "part" not in low:
                        fail("cut-off object not reported as partial")
                elif i["colour"] not in low:
                    fail(f"colour: expected {i['colour']}")
            elif len(vis) > 1:
                if "which one" not in low:
                    fail("several objects but did not ask which one")
            else:
                if "looks" in low and "when i saw it" not in low:
                    fail("claimed a live colour for an invisible object")
                if "can't see" not in low and "don't see" not in low:
                    fail("invisible object not said to be invisible")
        if kind == "SIDE" and not sc["stale"]:
            on_side = [i for i in items if i["side"] == side]
            if "anyone" in q.lower() or (cls == "person" and "is there a" in q.lower()):
                pass
            elif re.search(r"is there a \w+", q.lower()):
                truth = any(i["cls"] == cls for i in on_side)
                if truth != low.startswith("yes"):
                    fail(f"yes/no wrong for {cls} {side}")
            else:
                if not on_side and "don't see anything" not in low:
                    fail("listed things on an empty side")
                for c in OBJ:
                    if re.search(rf"\b(a|an|the|your) {c}\b", low) and not any(i["cls"] == c for i in on_side):
                        fail(f"listed {c} that is not {side}")
        if kind == "PAST":
            for row in mem.rows:
                spoken = "the " + row["name"]
                if spoken in low:
                    import spatial
                    rel = spatial.relative_angle(row["world_angle"], h.heading())
                    a_ = abs(rel)
                    d = "ahead" if a_ <= 35 else ("behind" if a_ > 135 else ("right" if rel > 0 else "left"))
                    if row["session_id"] != "S" or d != side:
                        fail(f"past: listed {row['name']} (dir {d}, session {row['session_id']})")
    return by_type


# ------------------------------------------------------------------ part B
def existing_phrases(rng):
    o, o2 = rng.choice(OBJ), rng.choice(OBJ)
    nm = rng.choice(KNOWN)
    return rng.choice([
        f"where is my {o}", f"where is the {o}", f"where's {nm}", f"where did I leave my {o}",
        f"where did I last see my {o}", "where is the door", "where are the stairs", f"how far is the {o}",
        "how far away is the door", f"how far is {nm}", f"find my {o}", f"help me find the {o}",
        "take me to the door", f"guide me to {nm}", f"look for my {o}", "stop looking", "i found it", "cancel",
        "never mind", f"this is {nm}", f"this is my {o}", f"my name is {nm}", f"call this my {o}",
        f"rename my {o} to my {o2}", f"change my {o} to {o2}", f"forget {nm}", f"forget my {o}", "be quiet",
        "quiet mode", "talk more", "normal mode", "speak slower", "speak faster", "slow down", "what changed",
        "what has changed in the room", "anything different", "what happened", "who was here", "who is here",
        "anyone here", "what did you see", "what have you seen recently", "what do you see", "what's around me",
        "describe the room", "tell me everything", "what is in the room", f"when did you see my {o}",
        f"when was {nm} here", f"what time did you see the {o}", f"where was my {o} before",
        f"where did my {o} used to be", "read this", "read the label", "keep reading", "stop reading",
        "what does this say", "read the text", f"is {nm}", f"listen, is {nm}", nm, f"where is {nm} now",
        "what is around me", "what's in the room", "who came"])


def part_b(rng, n, fails, lat):
    for case in range(n):
        now = 1_763_400_000.0 + case
        qa, h, mem, items, sc = make_scene(rng, now)
        q = noisy(rng, existing_phrases(rng), heavy=False)
        r = timed(qa, q, now, lat)
        if r is not None:
            fails.add("B", case, q, r.text, "Path B took an existing command")
        if h.spoken:
            fails.add("B", case, q, h.spoken, "Path B spoke during an existing command")
        if qa.stats["errors"]:
            fails.add("B", case, q, None, "error inside layer")


# ------------------------------------------------------------------ part C
def part_c(rng, n, fails, lat):
    for case in range(n):
        now = 1_763_500_000.0 + case * 100
        qa, h, mem, items, sc = make_scene(rng, now)
        cls = rng.choice(OBJ)
        first = rng.choice([f"what colour is the {cls}", f"how many {plural(cls)} are there",
                            f"is there a {cls} on my left", f"where is the {cls}"])
        timed(qa, first, now, lat)
        gap = rng.choice([2, 10, 30, 59, 61, 120])
        follow = rng.choice(["where is it", "where is it?", "when did you see it", "how far is it",
                             "where was it before"])
        r = timed(qa, follow, now + gap, lat)
        if gap <= 59:
            if r is None or r.kind != "rewrite" or cls not in r.text:
                fails.add("C", case, f"{first} -> (+{gap}s) {follow}", r and r.text, "follow-up not bound to subject")
        else:
            if r is not None:
                fails.add("C", case, f"{first} -> (+{gap}s) {follow}", r.text, "expired subject still used")
        if qa.stats["errors"]:
            fails.add("C", case, first, None, "error inside layer")


# ------------------------------------------------------------------ part D
def part_d(rng, n, fails):
    accepted = rejected = 0
    for case in range(n):
        now = 1_763_600_000.0 + case
        qa, h, mem, items, sc = make_scene(rng, now)
        facts = F.build(qa.snap.current, now, h)
        vis_lines = [l for l in facts.splitlines() if l.startswith("- ") and ":" in l]
        grounded = rng.random() < 0.5 and vis_lines
        if grounded:
            line = rng.choice(vis_lines)[2:]
            subj, rest = line.split(":", 1)
            where = rest.split(",")[0].strip()
            subj = re.sub(r"\s*\(.*\)", "", subj).strip()
            ans = rng.choice([f"I can see {subj} {where}.", f"{subj[0].upper() + subj[1:]} is {where}.",
                              "I don't know.", "I can't see that right now."])
            if "last seen" in rest:
                ans = f"I last saw {subj} {rest.split('last seen')[1].split(',')[0].strip()}."
            ok, why = G.check(G.clean(ans), facts)
            if not ok:
                fails.add("D", case, "grounded", ans, f"good answer rejected ({why})")
            accepted += ok
        else:
            missing_obj = [c for c in OBJ if c not in facts.lower()]
            missing_col = [c for c in BGR if c not in facts.lower()]
            lie = rng.choice([
                f"There is a {rng.choice(missing_obj)} in front of you." if missing_obj else "Rahul is here.",
                f"The {rng.choice(OBJ)} is {rng.choice(missing_col)}." if missing_col else "Rahul is here.",
                "Rahul is on your left.", "Priya is standing in front of you.", "The path ahead is clear.",
                "It is safe to walk forward.", f"I can see {rng.randint(13, 99)} things.",
                "Your keys are on the table.", "There are eleven chairs around you.",
                "Your wallet is next to the laptop."])
            ok, why = G.check(G.clean(lie), facts)
            if ok:
                fails.add("D", case, "hallucination", lie, "invented fact was NOT rejected")
            rejected += not ok
    return accepted, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    fails, lat = Failures(), []
    t0 = time.time()
    by_type = part_a(rng, 5000, fails, lat)
    part_b(rng, 3000, fails, lat)
    part_c(rng, 1000, fails, lat)
    acc, rej = part_d(rng, 1000, fails)
    lat_arr = np.array(lat)
    slow = int((lat_arr > MAX_MS).sum())
    summary = {"cases": 10000, "failed_checks": fails.count, "slow_answers_over_20ms": slow,
               "latency_ms": {"p50": round(float(np.percentile(lat_arr, 50)), 3),
                              "p99": round(float(np.percentile(lat_arr, 99)), 3),
                              "max": round(float(lat_arr.max()), 3)},
               "part_a_by_type": by_type, "part_d_grounded_accepted": int(acc), "part_d_lies_rejected": int(rej),
               "seconds": round(time.time() - t0, 1), "examples": fails.items}
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
    for f in fails.items[:15]:
        print("FAIL", f)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2)
    ok = fails.count == 0 and slow == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
