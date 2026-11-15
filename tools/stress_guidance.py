"""
tools/stress_guidance.py - 10,000 randomised "find my ..." sessions (guidance.py).

    python tools/stress_guidance.py [--iterations 10000] [--seed 0]

This is the guidance LOGIC only (no camera, YOLO or speech): a simulated user
stands somewhere, the item is at a random angle and distance, sometimes in
view, sometimes remembered, sometimes nowhere. Detections are noisy, missed
at random, occasionally wildly wrong, and hazard alerts block guidance at
random moments. The user follows the spoken cues imperfectly (overshoots,
slow reactions). Invariants checked on EVERY session:

  * no exception; the session always ends (reach, give-up or time limit)
  * no cue within MIN_GAP_S of the previous one (REACH_GAP_S for "within reach")
  * nothing spoken while speaking is blocked (hazard alert / SENSE talking)
  * at most one arrival message, and only when the item really is close
    (true distance <= 1.2 m) and roughly ahead (true angle <= 30 deg)
  * for an item that is NOT there: never "Found", always a give-up message
  * spoken words never say the opposite side of where the item truly is
    (checked for confirmed left/right cues with the item > 20 deg off-centre)
  * the guide's internal list stays bounded (recent_dist <= 3)

The full-app equivalent is tools/run_checks.py (find_item.mp4 scenario).
"""
import argparse
import os
import random
import sys
import time
import tracemalloc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import guidance  # noqa: E402
import spatial  # noqa: E402

FPS = 15.0
HALF_FOV = spatial.CAMERA_FOV_DEG / 2


def session(rng):
    kind = rng.choice(["visible", "memory", "nomemory", "absent", "moved"])
    item_angle = rng.uniform(-180, 180)
    dist = rng.uniform(1.0, 6.0)
    heading = 0.0
    if kind == "visible":
        item_angle = rng.uniform(-28, 28)
    miss = rng.uniform(0.0, 0.6)
    noise = rng.uniform(0.5, 4.0)
    outlier = rng.uniform(0.0, 0.05)
    block = rng.uniform(0.0, 0.3)          # share of time a hazard alert / speech blocks guidance
    react = rng.uniform(0.2, 1.0)          # user reaction delay (s)
    turn_speed = rng.uniform(15, 60)
    walk_speed = rng.uniform(0.3, 0.9)
    unknown_dist = rng.random() < 0.15     # a class without a size: distance never known
    present = kind != "absent"
    remembered = item_angle if kind == "memory" else (item_angle + rng.uniform(60, 120) if kind == "moved" else None)

    g = guidance.FindGuide()
    errs = []

    def sight(t):
        rel = spatial.wrap_angle(item_angle - heading)
        if not present or abs(rel) > HALF_FOV - 1 or rng.random() < miss:
            return []
        a = rel + rng.gauss(0, noise)
        if rng.random() < outlier:
            a = rng.uniform(-HALF_FOV, HALF_FOV)
        d = None if unknown_dist else dist * rng.uniform(0.85, 1.15)
        return [{"angle": a, "dist_m": d, "max_m": None, "filled": False}]

    if kind == "visible" and sight(0):
        first = g.start_visible("your bottle", 0.0, {"angle": spatial.wrap_angle(item_angle), "dist_m": dist,
                                                     "max_m": None}, heading)
    else:
        mem = None
        if remembered is not None:
            rel = spatial.relative_angle(remembered, heading)
            mem = {"world_angle": remembered, "age_text": "a minute ago",
                   "where": spatial.direction_phrase(rel), "relation": None,
                   "turn": spatial.turn_instruction(rel)}
        first = g.start("your bottle", 0.0, mem)
    said = [(0.0, first)]
    instruction, instr_at = None, 0.0

    def apply(text, t):
        nonlocal instruction, instr_at
        low = text.lower()
        new = None
        if "straight" in low or "should be in front" in low:
            new = "walk" if "straight" in low else "scan"
        elif "left" in low:
            new = "left"
        elif "right" in low:
            new = "right"
        elif "turn" in low:
            new = "scan"
        if new:
            instruction, instr_at = new, t + react

    apply(first, 0.0)
    last_cue_t = 0.0
    arrived = 0
    for i in range(1, int((guidance.TOTAL_TIMEOUT_S + 10) * FPS)):
        t = i / FPS
        if i % 3 == 0:
            g.observe(t, sight(t), heading)
        blocked = rng.random() < block
        out = g.poll(t, heading, can_speak=not blocked)
        if out:
            if blocked:
                errs.append(f"spoke while blocked: {out!r}")
            gap = t - last_cue_t
            need = guidance.REACH_GAP_S if ("within reach" in out or "right in front" in out) else guidance.MIN_GAP_S
            if gap < need - 1e-6 and len(said) > 1:
                errs.append(f"cues {gap:.2f}s apart: {said[-1][1]!r} -> {out!r}")
            rel = spatial.wrap_angle(item_angle - heading)
            if out.startswith(("A little", "On your", "Found")) and abs(rel) > 20 and present:
                side_said = "left" if "left" in out else ("right" if "right" in out else None)
                true_side = "right" if rel > 0 else "left"
                if side_said and side_said != true_side and abs(rel) < HALF_FOV - 5:
                    errs.append(f"wrong side: said {out!r}, item at {rel:.0f} deg")
            if "within reach" in out or "right in front" in out:
                arrived += 1
                if dist > 1.2 or abs(rel) > 30:
                    errs.append(f"arrival at {dist:.2f} m, {rel:.0f} deg: {out!r}")
            if out.startswith("Found") and not present:
                errs.append("found an item that is not there")
            said.append((t, out))
            last_cue_t = t
            apply(out, t)
        if not g.active:
            break
        if len(getattr(g, "recent_dist", [])) > 3:
            errs.append("recent_dist grew")
        # the user acts on the last instruction after a reaction delay
        if t >= instr_at:
            if instruction == "left":
                heading -= turn_speed / FPS
            elif instruction == "right":
                heading += turn_speed / FPS
            elif instruction == "scan":
                heading += 25.0 / FPS
            elif instruction == "walk":
                rel = spatial.wrap_angle(item_angle - heading)
                if abs(rel) < 35:
                    dist = max(0.35, dist - walk_speed / FPS)
                heading += rng.gauss(0, 0.8)          # walking is not perfectly straight
            heading = spatial.wrap_angle(heading)
            if instruction in ("left", "right") and abs(spatial.wrap_angle(item_angle - heading)) < 4:
                instruction = None                    # it is ahead now: wait for the next cue
    if g.active:
        errs.append(f"session never ended ({kind})")
    if arrived > 1:
        errs.append(f"{arrived} arrival messages")
    if not present and not any("couldn't find" in s or "stopping the search" in s for _, s in said):
        errs.append(f"absent item: no give-up message: {said[-2:]}")
    outcome = "arrived" if arrived else ("ended, size unknown" if unknown_dist and present else "gave up")
    return kind, outcome, errs, len(said), t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tracemalloc.start()
    m0 = tracemalloc.get_traced_memory()[0]
    t0 = time.time()
    fails, outcomes, cues, sim_s = [], {}, 0, 0.0
    per_k = []
    for k in range(a.iterations):
        seed = a.seed * 1_000_003 + k
        ts = time.perf_counter()
        try:
            kind, outcome, errs, n, dur = session(random.Random(seed))
        except Exception as e:  # noqa: BLE001
            kind, outcome, errs, n, dur = "?", "crash", [f"exception {e!r}"], 0, 0.0
        per_k.append(time.perf_counter() - ts)
        outcomes[(kind, outcome)] = outcomes.get((kind, outcome), 0) + 1
        cues += n
        sim_s += dur
        if errs:
            fails.append((seed, kind, errs[:3]))
    wall = time.time() - t0
    m1 = tracemalloc.get_traced_memory()[0]
    first, last = per_k[:1000], per_k[-1000:]
    print(f"{a.iterations} guidance sessions, {sim_s / 3600:.1f} simulated hours, {cues} sentences, "
          f"{wall:.1f} s wall time")
    print(f"time per session: first 1000 avg {1000 * sum(first) / len(first):.2f} ms, "
          f"last 1000 avg {1000 * sum(last) / len(last):.2f} ms")
    print(f"python memory: {m0 / 1e6:.2f} MB -> {m1 / 1e6:.2f} MB")
    for (kind, outcome), n in sorted(outcomes.items()):
        print(f"  {kind:9s} {outcome:20s} {n}")
    print(f"FAILED sessions: {len(fails)}")
    for seed, kind, errs in fails[:15]:
        print(f"  seed {seed} ({kind}): {errs}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
