"""
tools/stress_personal.py - 10,000 random rooms: "my cup" must never be given
to a different physical object.

    python tools/stress_personal.py            # 10,000 scenarios
    python tools/stress_personal.py 500 --seed 7

Each scenario builds a room (tests/personal_sim.py) with 1-2 personal items
and 1-3 other objects of the same kind - often IDENTICAL in colour (the
hardest case) - then does random things for a while:

    turn the camera (sometimes 45 degrees, sometimes all the way round),
    wait, let ByteTrack swap two ids during a turn, add heading error,
    take the personal item away, restart SENSE (new run, same database),
    ask "where is my ...?"

Checked on every frame and every answer:
    SAFETY   a personal name is never shown on, or stored for, another object
    HONESTY  while the item is out of view the answer never says it is seen,
             and the stored place stays the item's last real place
    QUESTION "where is my <kind>" resolves to that personal item

Reported (not required to be 100 %): how often the real item is recognised
again when it comes back into view (the price of being careful).

Objects of one kind are placed >= 45 degrees apart, like the reported bug
(turn 45 degrees, see another cup). Not covered, because no camera-only
system can tell: an identical object put exactly where yours was, after
yours was taken away without SENSE seeing it go.
"""

import argparse
import os
import sqlite3
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import contextlib  # noqa: E402
import io  # noqa: E402

from personal_sim import LOOKS, Room, chroma_signature, render  # noqa: E402

import personal  # noqa: E402
import spatial  # noqa: E402

CATEGORIES = ["cup", "bottle", "phone", "laptop", "bag", "chair"]
ALL_CATS = set(CATEGORIES) | {"remote", "book", "mouse"}


def build_pool(rng, per_look=60):
    pool = {}
    for look in LOOKS:
        pool[look] = [chroma_signature(render(look, rng, 1.0 + rng.uniform(-0.1, 0.1),
                                              tuple(1.0 + rng.uniform(-0.03, 0.03, 3))))
                      for _ in range(per_look)]
    return pool


def spread_angles(rng, n, min_gap=45.0):
    for _ in range(200):
        angles = sorted(float(a) for a in rng.uniform(-180, 180, n))
        gaps = [(angles[(i + 1) % n] - angles[i]) % 360 for i in range(n)] if n > 1 else [360]
        if min(gaps) >= min_gap:
            return angles
    return [(-180 + i * 360.0 / n) for i in range(n)]


def scenario(seed, pool, stats):
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(":memory:")
    session = [0]

    def new_target():
        session[0] += 1
        with contextlib.redirect_stdout(io.StringIO()):
            return personal.create(f"run{session[0]}", None, ALL_CATS, conn=conn)

    p = new_target()
    room = Room(p, seed=seed, heading_noise=float(rng.uniform(0, 4)), pool=pool)
    looks = list(LOOKS)

    # --- the room ------------------------------------------------------------
    n_items = 1 if rng.random() < 0.7 else 2
    same_kind = n_items == 2 and rng.random() < 0.3
    cats = [str(rng.choice(CATEGORIES))]
    if n_items == 2:
        cats.append(cats[0] if same_kind else str(rng.choice([c for c in CATEGORIES if c != cats[0]])))
    personal_keys = {}
    for cat in dict.fromkeys(cats):
        k_items = cats.count(cat)
        n_other = int(rng.integers(1, 4))
        angles = spread_angles(rng, k_items + n_other)
        base_look = str(rng.choice(looks))
        for j, a in enumerate(angles):
            if j < k_items:
                look = base_look if j == 0 else str(rng.choice([l for l in looks if l != base_look]))
                key = f"{cat}-mine{j}"
                first = f"my {cat}" if rng.random() < 0.5 else f"my work {cat}"
                personal_keys[key] = first if j == 0 else f"my second {cat}"
            else:
                # half of the others look IDENTICAL to the (first) personal one
                look = base_look if rng.random() < 0.5 else str(rng.choice(looks))
                key = f"{cat}-other{j}"
            room.add(key, cat, look, a)

    # --- "this is my cup" -------------------------------------------------------
    for key, name in personal_keys.items():
        o = next(x for x in room.objects if x.key == key)
        room.turn_to(o.angle + float(rng.uniform(-8, 8)))
        room.run(4)
        if key not in room.track_of:
            continue
        room.teach(name, key)
        room.run(int(rng.integers(3, 15)))

    truth = {name: key for key, name in personal_keys.items()}
    last_true_place = {}

    def check_frame(seen):
        for key, label in seen.items():
            if label and truth.get(label) != key:
                stats["wrong_frames"] += 1
                stats.setdefault("examples", []).append((seed, label, key))
        for name, key in truth.items():
            item = room.target.matcher.items.get(name) if room.target.enabled else None
            if item is None:
                continue
            if key in seen:
                stats["return_frames"] += 1
                stats["return_hits"] += seen[key] == name

    # --- random life -------------------------------------------------------------
    for _ in range(int(rng.integers(8, 16))):
        r = rng.random()
        if r < 0.45:                       # turn somewhere
            if rng.random() < 0.5:
                target = float(rng.choice([o.angle for o in room.objects]))
            else:
                target = room.heading + float(rng.choice([45, -45, 90, -90, 180]))
            target += float(rng.uniform(-10, 10))
            switch = None
            vis = [o for o, _ in room.visible()]
            if rng.random() < 0.15 and len(vis) >= 2:
                same = [(a.key, b.key) for a in vis for b in vis if a.key < b.key and a.category == b.category]
                if same:
                    switch = same[int(rng.integers(len(same)))]
            room.turn_to(target)
            if switch:
                room.step(stable=False, track_switch=switch)
                room.now += 0.9
            room.drift = float(np.clip(room.drift + rng.normal(0, 3), -10, 10))
        elif r < 0.75:                     # look for a while
            for _ in range(int(rng.integers(5, 30))):
                check_frame(room.step())
        elif r < 0.82:                     # someone takes my item away
            key = str(rng.choice(list(truth.values())))
            next(o for o in room.objects if o.key == key).present = False
        elif r < 0.90:                     # restart SENSE
            room.target.flush()
            room.target = new_target()
            room.track_of = {}
            if rng.random() < 0.5:
                room.drift = float(rng.uniform(-5, 5))     # restarted facing the same way
            else:                                          # the heading starts again at 0
                room.drift = spatial.wrap_angle(-room.heading + float(rng.uniform(-5, 5)))
            stats["restarts"] += 1
        else:                              # ask
            for name, key in truth.items():
                stats["questions"] += 1
                cat = name.split()[-1]
                # the exact name always works; "my <kind>" works when it is unambiguous
                ok = room.target.query(f"where is {name}?") == ("target", name)
                q = room.target.query(f"where is my {cat}?")
                kin = [n for n in truth if n.split()[-1] == cat]
                if f"my {cat}" in truth:
                    ok = ok and q == ("target", f"my {cat}")
                elif len(kin) > 1:
                    ok = ok and q is not None and q[0] == "say" and "Do you mean" in q[1]
                else:
                    ok = ok and q == ("target", name)
                if not ok:
                    stats["bad_questions"] += 1
                    stats.setdefault("examples", []).append((seed, "question", name, q))
                st = room.target.matcher.status(name, room.now)
                vis_keys = {o.key for o, _ in room.visible()}
                # (st["visible"] can only come from a track locked to this item,
                # and SAFETY above checks that every such track IS the item)
                note = room.target.note(name, room.now, room.heading)
                if key not in vis_keys and room.now - (st["item"].last_seen or 0) > 2.5 \
                        and "can't see it" not in note:
                    stats["dishonest"] += 1      # out of view, but not said so
                    stats.setdefault("examples", []).append((seed, "honesty", name, note, st["visible"]))
    check_frame(room.step())
    # the stored place of each item must be its own (never another object's)
    for name, key in truth.items():
        item = room.target.matcher.items.get(name)
        o = next(x for x in room.objects if x.key == key)
        if item is not None and item.world_angle is not None and item.session_id is not None:
            err = abs(spatial.wrap_angle(item.world_angle - o.angle))
            stats["place_checks"] += 1
            if err > 25 + 15:          # near_deg + heading noise/drift
                stats["wrong_place"] += 1
                stats.setdefault("examples", []).append((seed, "place", name, round(err, 1)))
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(12345)
    pool = build_pool(rng)
    stats = {k: 0 for k in ["wrong_frames", "return_frames", "return_hits", "restarts", "questions",
                            "bad_questions", "dishonest", "place_checks", "wrong_place"]}
    t0 = time.time()
    for k in range(a.n):
        with contextlib.redirect_stdout(io.StringIO()):
            scenario(a.seed * 1_000_003 + k, pool, stats)
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1:,} scenarios, {time.time() - t0:.0f} s, wrong so far: {stats['wrong_frames']}")
    ok = (stats["wrong_frames"] == 0 and stats["bad_questions"] == 0 and stats["dishonest"] == 0
          and stats["wrong_place"] == 0)
    rate = stats["return_hits"] / max(1, stats["return_frames"])
    print(f"Scenarios: {a.n:,}   restarts: {stats['restarts']:,}   questions: {stats['questions']:,}")
    print(f"SAFETY   frames where a personal name was on another object: {stats['wrong_frames']}")
    print(f"PLACE    stored places that belong to another object: {stats['wrong_place']} "
          f"of {stats['place_checks']:,}")
    print(f"QUESTION wrong resolutions: {stats['bad_questions']}   HONESTY violations: {stats['dishonest']}")
    print(f"Recognised again when in view (not required to be 100%): {rate:.1%} "
          f"of {stats['return_frames']:,} frames")
    for ex in stats.get("examples", [])[:10]:
        print("  example:", ex)
    print(f"RESULT: {'PASS' if ok else 'FAIL'} ({a.n:,} cases)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
