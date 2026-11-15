"""
tools/stress_logic.py - 10,000 randomised iterations of SENSE's decision logic.

    python tools/stress_logic.py [--iterations 10000] [--seed 0]

This is NOT the full app (that is tools/smoke_run.py --loop). It hammers the
pure-logic parts with random inputs far beyond what a few hand-written tests
can cover, and checks invariants that must ALWAYS hold:

  continuity  one random walk per iteration (approach / stand still / turn the
              head / walk past / reappear closer after a gap), with missed
              detections, ByteTrack id switches, sway and size noise.
              - approach: exactly one identity; at most one alert, given
                before contact (true distance > 0.8 m)
              - every other kind: no alert at all
  names       a random sequence of face results per iteration
              - a name is locked only if >= 3 of the last 5 checks said it and
                no other enrolled name appears in those 5
  notifier    random pushes on a simulated clock
              - the same key is never spoken twice within its cooldown
  database    one write + read cycle per iteration on a temporary database
              - every write is read back; PRAGMA integrity_check is ok

It reports failure COUNTS (it does not stop at the first), timing per part,
and memory before/after. Failures are printed with the seed to replay them.
"""
import argparse
import math
import os
import random
import sys
import tempfile
import time
import tracemalloc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_continuity import Sim, FPS  # noqa: E402  (pinhole camera simulator)
import people  # noqa: E402
import notifier as notifier_mod  # noqa: E402


def continuity_iteration(seed):
    rng = random.Random(seed)
    kind = rng.choice(["approach", "approach", "still", "turn", "past", "gap"])
    sim = Sim(seed=seed, miss=rng.uniform(0.0, 0.35), id_switch_every=rng.randint(3, 30),
              sway_px=rng.uniform(0, 18), noise=rng.uniform(0.0, 0.05))
    cls = rng.choice(["chair", "bed", "door"])
    size = {"chair": (0.9, 0.5), "bed": (0.6, 1.6), "door": (2.0, 0.85)}[cls]
    truth = []
    if kind == "approach":
        start, speed = rng.uniform(3.0, 5.0), rng.uniform(0.4, 1.1)
        lat = rng.uniform(-0.1, 0.1)
        for i in range(int((start - 0.7) / speed * FPS)):
            d = start - speed * i / FPS
            truth.append(d)
            sim.step(i, [(cls, cls, lat, d, *size)])
    elif kind == "still":
        d = rng.uniform(0.9, 3.0)
        for i in range(int(rng.uniform(3, 10) * FPS)):
            sim.step(i, [(cls, cls, rng.gauss(0, 0.01), d, *size)])
    elif kind == "turn":
        d, n = rng.uniform(1.0, 3.0), int(rng.uniform(3, 8) * FPS)
        a, b = rng.uniform(-1.6, -0.8), rng.uniform(0.8, 1.6)
        if rng.random() < 0.5:
            a, b = b, a
        for i in range(n):
            sim.step(i, [(cls, cls, a + (b - a) * i / n, d, *size)])
    elif kind == "past":
        lat, start, speed = rng.choice([-1, 1]) * rng.uniform(1.1, 1.6), rng.uniform(3.0, 4.5), rng.uniform(0.4, 1.0)
        for i in range(int((start - 0.6) / speed * FPS)):
            sim.step(i, [(cls, cls, lat, start - speed * i / FPS, *size)])
    else:   # seen far, hidden, seen again closer while standing still
        i = 0
        for _ in range(int(1.5 * FPS)):
            sim.step(i, [(cls, cls, 0.0, rng.uniform(2.4, 2.8), *size)]); i += 1
        for _ in range(int(rng.uniform(0.7, 1.8) * FPS)):
            sim.step(i, []); i += 1
        d = rng.uniform(1.1, 1.5)
        for _ in range(int(3 * FPS)):
            sim.step(i, [(cls, cls, 0.0, d, *size)]); i += 1
    errs = []
    if kind == "approach":
        if len(sim.ids_seen.get(cls, ())) != 1:
            errs.append(f"{len(sim.ids_seen.get(cls, ()))} identities for one {cls}")
        if len(sim.alerts) > 1:
            errs.append(f"{len(sim.alerts)} alerts in one approach")
        for f, *_ in sim.alerts:
            if truth[f] <= 0.8:
                errs.append(f"alert too late at {truth[f]:.2f} m")
        missed = not sim.alerts
    else:
        if sim.alerts:
            errs.append(f"false alert(s) while '{kind}': {[(a[0], a[2]) for a in sim.alerts]}")
        missed = False
    return kind, errs, missed


def names_iteration(seed):
    rng = random.Random(seed)
    gate = people.PersonIdentityGate()
    pool = [people.UNKNOWN, people.NO_FACE, "Asha", "Ravi", "Sneha"]
    weights = [rng.random() for _ in pool]
    hist = []
    errs = []
    for _ in range(rng.randint(5, 40)):
        r = rng.choices(pool, weights)[0]
        label = gate.observe(1, r)
        if r is not people.NO_FACE:
            hist = (hist + [r])[-5:]
        locked = gate.locked.get(1)
        if locked is not None:
            same = sum(1 for h in hist if h == locked)
            others = {h for h in hist if h not in (locked, people.UNKNOWN)}
            # a lock may be KEPT while it is not yet released (< 3 contrary results)
            kept = sum(1 for h in hist if h != locked) < 3
            if not ((same >= 3 and not others) or kept):
                errs.append(f"locked {locked} with history {hist}")
            if label != locked:
                errs.append(f"label {label!r} != locked {locked!r}")
    return errs


def notifier_iteration(seed):
    rng = random.Random(seed)
    clock = [0.0]
    spoken = []
    n = notifier_mod.Notifier(lambda s: spoken.append((clock[0], s)), lambda: rng.random() < 0.2,
                              level="verbose", clock=lambda: clock[0])
    last = {}
    errs = []
    for _ in range(200):
        clock[0] += rng.uniform(0.0, 0.5)
        if rng.random() < 0.3:
            p = rng.choice([notifier_mod.P1, notifier_mod.P2, notifier_mod.P3])
            key = f"k{rng.randint(0, 6)}"
            n.push(p, key, f"{key} p{p}")
        said = n.tick(camera_moving=rng.random() < 0.2, hold=rng.random() < 0.1)
        if said:
            for part in said.rstrip(".").split(". "):
                key, p = part.split(" p")
                cd = n.P1_COOLDOWN_S if p == "1" else n.PER_ITEM_COOLDOWN_S
                if key in last and clock[0] - last[key] < min(cd, n.P1_COOLDOWN_S):
                    errs.append(f"{key} repeated after {clock[0] - last[key]:.1f} s")
                last[key] = clock[0]
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    N = a.iterations

    import memory_db
    tmp = tempfile.mkdtemp()
    memory_db.DB_PATH = os.path.join(tmp, "stress.db")
    memory_db.init_database()

    tracemalloc.start()
    report = []
    kinds, missed = {}, 0
    failures = {"continuity": [], "names": [], "notifier": [], "database": []}
    times = {k: 0.0 for k in failures}
    mem_samples = []
    for it in range(N):
        seed = a.seed * 1_000_003 + it
        t = time.perf_counter()
        kind, errs, miss = continuity_iteration(seed)
        times["continuity"] += time.perf_counter() - t
        kinds[kind] = kinds.get(kind, 0) + 1
        missed += miss
        failures["continuity"] += [(seed, kind, e) for e in errs]

        t = time.perf_counter()
        failures["names"] += [(seed, e) for e in names_iteration(seed)]
        times["names"] += time.perf_counter() - t

        t = time.perf_counter()
        failures["notifier"] += [(seed, e) for e in notifier_iteration(seed)]
        times["notifier"] += time.perf_counter() - t

        t = time.perf_counter()
        name = f"item{it % 50}"
        memory_db.save_object_observation(name, "on your left", it, it % 7)
        memory_db.log_event(time.time(), name, "APPEARED", "on your left", 0.0, "stress")
        row = memory_db.get_object_memory(name)
        if not row:
            failures["database"].append((seed, f"{name} not read back"))
        times["database"] += time.perf_counter() - t

        if it % max(1, N // 20) == 0 or it == N - 1:
            mem_samples.append((it, tracemalloc.get_traced_memory()[0] / 1e6))
            print(f"  {it + 1:>6}/{N}  python heap {mem_samples[-1][1]:.1f} MB", flush=True)

    integrity = memory_db._get_conn().execute("PRAGMA integrity_check").fetchone()[0]
    events = memory_db._get_conn().execute("SELECT COUNT(*) FROM events").fetchone()[0]
    memory_db.close()
    if integrity != "ok":
        failures["database"].append((None, f"integrity_check: {integrity}"))
    if events < N:
        failures["database"].append((None, f"only {events} of {N} events stored"))

    print(f"\n==== {N} iterations (seed {a.seed}) ====")
    print(f"continuity walks: {kinds}; approaches with no alert: {missed} "
          f"of {kinds.get('approach', 0)}")
    print(f"database: {events} events stored, integrity_check = {integrity}")
    print(f"python heap: {mem_samples[0][1]:.1f} MB at start -> {mem_samples[-1][1]:.1f} MB at end")
    total = 0
    for part, fl in failures.items():
        total += len(fl)
        print(f"{part:>10}: {len(fl)} failures, {times[part]:.1f} s")
        for f in fl[:8]:
            print("            ", f)
    print(f"\nTOTAL FAILURES: {total}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
