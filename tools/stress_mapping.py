"""
tools/stress_mapping.py - 10,000 random spoken sentences for the Phase 0 home map.

    python tools/stress_mapping.py            # 10,000 cases
    python tools/stress_mapping.py 2000 --seed 3

Three kinds of cases, mixed at random:

  NEW       a room command in many wordings, with Whisper-style noise
            (capitals, punctuation, "okay sense ...", "bed room", trailing "please").
            Must be recognised as the right command and the right room.
  EXISTING  a sentence SENSE already understands ("this is Sneha", "take me to
            my phone", "what's in this room", ...). Must return None, so the
            existing voice code receives it exactly as before.
  SESSION   a random conversation (teach rooms, ask where am I, take me to,
            list) run through HomeMap on an in-memory database and compared
            with a simple reference model of what the answer must say.
            (Teaching needs camera frames, so these conversations cancel it.)
  TEACH     (Phase 1) a random head-turn while teaching: any speed, either
            direction, pauses, back-and-forth, blurred frames, "stop" early.
            Never more than one view per 20-degree direction, never more
            than 18, always ends by the time limit, "done" only with enough views.
  RECOGNISE (Phase 1) a random home of 2-8 taught rooms (synthetic image
            fingerprints with realistic noise) and a random walk through the
            rooms and untaught places, with odd frames (10% junk such as
            blur, 3% a clear glimpse of another place). From the 3rd check
            after a doorway, "You are in X" must never name the wrong room;
            "I think" may, and is counted.
"""

import argparse
import os
import random
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                   # noqa: E402

import mapping                       # noqa: E402
from mapping.config import MappingConfig       # noqa: E402
from mapping.recognise import UNKNOWN, Recogniser, RoomModel  # noqa: E402
from mapping.teaching import TeachSession      # noqa: E402
from mapping import phrases as P     # noqa: E402
from mapping.config import ROOM_WORDS  # noqa: E402

ROOMS = ["kitchen", "bedroom", "bathroom", "living room", "hall", "study", "balcony",
         "dining room", "guest room", "pooja room", "store room", "sneha's room", "my room"]
SPLIT = {"bedroom": "bed room", "bathroom": "bath room", "living room": "livingroom"}
PEOPLE = ["sneha", "harman", "krish", "mom", "rahul"]
OBJECTS = ["phone", "remote", "cup", "bottle", "book", "laptop", "mouse", "bag", "chair",
           "couch", "bed", "table", "plant", "tv", "door", "stairs"]
LEAD = ["", "", "", "okay ", "sense ", "hey sense ", "please ", "um ", "so "]
TRAIL = ["", "", "", " please", " now", ".", "?", "!", " sense"]

TEACH_T = ["this room is the {r}", "this room is {r}", "call this room the {r}", "name this room {r}",
           "this room is called the {r}", "remember this room as the {r}", "save this room as {r}"]
WHERE_T = ["where am i", "which room am i in", "what room am i in", "what room is this",
           "which room is this", "where are we", "where am i right now", "which room are we in"]
GO_T = ["take me to the {r}", "take me to {r}", "guide me to the {r}", "how do i get to the {r}",
        "lead me to the {r}", "bring me to the {r}", "take me back to the {r}", "help me get to the {r}"]
LIST_T = ["which rooms do you know", "what rooms do you know", "list the rooms", "list all rooms",
          "how many rooms do you know", "what rooms have you learned", "which rooms did i teach you"]

EXISTING_T = [
    "this is {p}", "is {p}", "this is my {o}", "this is the {o}", "this is a {o}", "my name is {p}",
    "call this my {o}", "remember this as my {o}", "take me to my {o}", "take me to the {o}",
    "take me to {p}", "guide me to the {o}", "find my {o}", "find the {o}", "find {p}",
    "where is my {o}", "where is the {o}", "where is {p}", "where was my {o}", "when did you see {p}",
    "what's in this room", "what is in the room", "describe the room", "what do you see",
    "how many people are in the room", "what colour is the {o}", "how many {o}s are there",
    "what time is it", "forget my {o}", "forget {p}", "rename my {o} to my mug", "what changed",
    "who is here", "read this", "keep reading", "stop looking", "be quiet", "speak slower",
    "what did you see recently", "is there anything on my right", "what is in front of me",
    "this is the {r}", "it is the {r}", "i am in the {r}", "find the {r}", "forget the {r}",
    "what is next to the {r}", "where is the {r}", "room", "rooms", "the room",
]


def noisy(rng, s):
    s = rng.choice(LEAD) + s + rng.choice(TRAIL)
    r = rng.random()
    if r < 0.3:
        s = s.capitalize()
    elif r < 0.4:
        s = s.upper()
    return s.replace(" ", "  ") if rng.random() < 0.05 else s


def say_room(rng, room):
    return SPLIT[room] if room in SPLIT and rng.random() < 0.4 else room


def new_case(rng):
    kind = rng.choice([P.TEACH, P.WHERE_AM_I, P.GO, P.LIST])
    room = rng.choice(ROOMS)
    tmpl = {P.TEACH: TEACH_T, P.WHERE_AM_I: WHERE_T, P.GO: GO_T, P.LIST: LIST_T}[kind]
    t = rng.choice(tmpl)
    if "{r}" in t and room.startswith(("sneha's", "my ")):
        t = t.replace("the {r}", "{r}")
    text = noisy(rng, t.format(r=say_room(rng, room)))
    cmd = P.parse(text, known_rooms=(), known_names=set(PEOPLE))
    want_room = P.normalise(room) if kind in (P.TEACH, P.GO) else None   # "store room" is saved as "storeroom"
    ok = cmd is not None and cmd.kind == kind and cmd.room == want_room
    return ok, f"NEW {kind}: {text!r} -> {cmd}"


def existing_case(rng):
    t = rng.choice(EXISTING_T)
    text = noisy(rng, t.format(p=rng.choice(PEOPLE), o=rng.choice(OBJECTS), r=rng.choice(ROOMS)))
    known_rooms = rng.sample(ROOMS, rng.randint(0, 4))
    cmd = P.parse(text, known_rooms=known_rooms, known_names=set(PEOPLE) | {"my bottle", "my cup"})
    return cmd is None, f"EXISTING: {text!r} (rooms {known_rooms}) -> {cmd}"


class Clock:
    t = 1_000_000.0

    def __call__(self):
        return self.t


def session_case(rng):
    clock = Clock()
    run = f"run{rng.randint(1, 3)}"
    home = mapping.HomeMap(sqlite3.connect(":memory:"), run, clock=clock)
    rooms, current, declared_at = [], None, None
    for _ in range(rng.randint(3, 12)):
        clock.t += rng.choice([1, 30, 300, 1200, 2400])
        action = rng.choice(["teach", "where", "go", "list"])
        room = rng.choice(ROOMS[:8])
        if action == "teach":
            text = rng.choice(TEACH_T).format(r=room)
            reply = home.handle(noisy(rng, text), clock())
            if room not in rooms:
                rooms.append(room)
            current, declared_at = room, clock()
            if room not in reply:
                return False, f"SESSION teach {room}: {reply!r}"
            if "turn slowly" in reply:            # no camera here: cancel the teaching
                reply = home.handle(rng.choice(["cancel", "never mind", "Cancel."]), clock())
                if "Nothing new was saved" not in reply:
                    return False, f"SESSION cancel: {reply!r}"
        elif action == "where":
            reply = home.handle(noisy(rng, rng.choice(WHERE_T)), clock())
            fresh = current is not None and clock() - declared_at <= 30 * 60
            if fresh and f"was the {current}" not in reply:
                return False, f"SESSION where (want {current}): {reply!r}"
            if not fresh and not reply.startswith("I don't know which room"):
                return False, f"SESSION where (want unknown): {reply!r}"
        elif action == "go":
            reply = home.handle(noisy(rng, rng.choice(GO_T).format(r=room)), clock())
            fresh = current is not None and clock() - declared_at <= 30 * 60
            if fresh and current == room:
                want = "already in"
            elif room in rooms:
                want = "can't guide you between rooms yet"
            else:
                want = "yet. When you are in it"
            if reply is None or want not in reply:
                return False, f"SESSION go {room} (rooms {rooms}, current {current}): {reply!r}"
        else:
            reply = home.handle(noisy(rng, rng.choice(LIST_T)), clock())
            if not rooms and "No rooms are registered yet" not in reply:
                return False, f"SESSION list empty: {reply!r}"
            if rooms and not all(r in reply for r in rooms):
                return False, f"SESSION list {rooms}: {reply!r}"
    if home.rooms.names() != rooms:
        return False, f"SESSION stored {home.rooms.names()} != {rooms}"
    return True, ""


CFG = MappingConfig()


def teach_case(rng):
    start = rng.uniform(-180, 180)
    s = TeachSession(1, "kitchen", start, 0.0, CFG)
    fps = rng.choice([8, 10, 15, 30])
    h, t, speed = start, 0.0, rng.uniform(-60, 60)
    blur_p = rng.choice([0.0, 0.1, 0.5, 0.95])
    stop_at = rng.uniform(1, 40) if rng.random() < 0.2 else None
    end = None
    while t < CFG.max_teach_s + 5:
        t += 1.0 / fps
        if rng.random() < 0.03:
            speed = rng.choice([0.0, rng.uniform(-60, 60), rng.uniform(5, 40)])
        h = ((h + speed / fps + rng.gauss(0, 0.5)) + 180) % 360 - 180
        if stop_at is not None and t >= stop_at:
            end = s.stop()
            break
        ev = s.feed(h, t, 0.0 if rng.random() < blur_p else 40.0,
                    lambda: {"emb": None, "pts": None, "des": None, "landmarks": []})
        if ev and ev[0] in ("done", "failed"):
            end = ev
            break
    msg = f"TEACH fps={fps} blur={blur_p} stop={stop_at} -> {end} {s!r}"
    if end is None:
        return False, msg + " (never ended)"
    bins = [v["bin"] for v in s.views]
    if len(bins) != len(set(bins)) or len(s.views) > s.bins_total:
        return False, msg + " (duplicate direction)"
    need = CFG.min_stop_views if stop_at is not None else min(CFG.min_views, CFG.done_bins)
    if end[0] == "done" and len(s.views) < need:
        return False, msg + " (done with too few views)"
    if end[0] == "failed" and stop_at is None and len(s.views) >= CFG.done_bins:
        return False, msg + " (failed with enough views)"
    return True, ""


def unit(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


RECOG = {"checks": 0, "sure_right": 0, "think_right": 0, "think_wrong": 0, "unsure": 0}


def recognise_case(rng):
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    dim, n_rooms = 128, rng.randint(2, 8)
    protos = [unit(nrng.normal(size=dim)) for _ in range(n_rooms + 3)]    # +3 untaught places
    shared = unit(nrng.normal(size=dim))       # what all indoor views share (walls, light)
    noise = rng.uniform(0.6, 1.1)              # how different two views of one room are

    def view(room):
        return unit(1.2 * shared + protos[room] + noise * unit(nrng.normal(size=dim)))
    rec = Recogniser(CFG)
    rec.set_rooms([RoomModel(i, f"r{i}", [{"emb": view(i), "pts": None, "des": None, "landmarks": []}
                                          for _ in range(rng.randint(8, 18))]) for i in range(n_rooms)])
    where, settled = rng.randrange(n_rooms + 3), 99
    for step in range(rng.randint(10, 60)):
        if rng.random() < 0.08:
            where, settled = rng.randrange(n_rooms + 3), 0    # walked somewhere else
        settled += 1
        if settled <= 2:                                      # in the doorway: a bit of both
            seen = rng.choice([where, rng.randrange(n_rooms + 3), "junk"])
        else:
            u = rng.random()      # odd frames: 10% junk (blur, a hand, the ceiling), 3% a clear
            seen = "junk" if u < 0.10 else rng.randrange(n_rooms + 3) if u < 0.13 else where   # glimpse elsewhere
        emb = unit(nrng.normal(size=dim)) if seen == "junk" else view(seen)
        rec.update(emb, None, None, set(), step)
        if settled < 5:
            continue          # judged from the 3rd check after the doorway (~2 s at one check per 0.7 s)
        state, p = rec.best()
        RECOG["checks"] += 1
        named = state is not None and state != UNKNOWN and p >= CFG.likely_prob
        sure = named and p >= CFG.sure_prob and rec.verified(state)
        truth = where if where < n_rooms else None
        if sure and state != truth:
            return False, (f"RECOGNISE: said 'You are in r{state}' (p={p:.2f}) but the user is in "
                           f"{'r%d' % truth if truth is not None else 'an untaught place'}; rooms={n_rooms} noise={noise:.2f}")
        if sure:
            RECOG["sure_right"] += 1
        elif named and state == truth:
            RECOG["think_right"] += 1
        elif named:
            RECOG["think_wrong"] += 1
        else:
            RECOG["unsure"] += 1
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    counts, fails = {"new": 0, "existing": 0, "session": 0, "teach": 0, "recognise": 0}, []
    cases = {"new": new_case, "existing": existing_case, "session": session_case,
             "teach": teach_case, "recognise": recognise_case}
    for _ in range(a.n):
        r = rng.random()
        kind = ("new" if r < 0.30 else "existing" if r < 0.55 else "session" if r < 0.65
                else "teach" if r < 0.80 else "recognise")
        ok, msg = cases[kind](rng)
        counts[kind] += 1
        if not ok:
            fails.append(msg)
    assert all(w in ROOM_WORDS for w in ("kitchen", "bedroom", "hall"))
    print(f"{a.n} cases (seed {a.seed}): {counts['new']} new commands, "
          f"{counts['existing']} existing commands, {counts['session']} conversations, "
          f"{counts['teach']} teaching turns, {counts['recognise']} homes walked through")
    c = RECOG
    if c["checks"]:
        print(f"recognition checks: {c['checks']}: 'You are in' right {c['sure_right']}, "
              f"'I think' right {c['think_right']}, 'I think' wrong {c['think_wrong']}, "
              f"not sure {c['unsure']} (a confident wrong answer fails the test)")
    print(f"{len(fails)} failed")
    for m in fails[:25]:
        print("  " + m)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
