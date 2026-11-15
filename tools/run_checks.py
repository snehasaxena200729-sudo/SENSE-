"""
tools/run_checks.py - every automated check, in one command, with PASS/FAIL.

    python tools/run_checks.py                 # unit + regression tests only
    python tools/run_checks.py --videos DIR    # + full-app scenarios on clips
                                               #   from tools/make_test_videos.py

Unit / regression: everything in tests/ (pure logic, no camera).
Scenarios: the REAL sense_master.py loop runs headless on each clip through
tools/smoke_run.py (same YOLO, ByteTrack, face, memory, notifier code), and
what SENSE would have said is checked against what it should say.

What this does NOT cover - see the physical test guide: the real webcam, the
microphone and Whisper, the `say` voice, walking with the camera, lighting.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def said(run, pattern):
    """Sentences SENSE spoke (not the scripted questions) matching a regex."""
    return [t for _, t in run["spoken"] if not t.startswith("[asked:") and re.search(pattern, t, re.I)]


def answer_to(run, question):
    """What SENSE said right after a scripted question."""
    sp = run["spoken"]
    for k, (_, t) in enumerate(sp):
        if t == f"[asked: {question}]":
            return sp[k + 1][1] if k + 1 < len(sp) else ""
    return None


def no_errors(run):
    return [] if not run["error"] else ["exception: " + run["error"].strip().splitlines()[-1]]


def check_approach(run):
    errs = no_errors(run)
    alerts = said(run, r"^Careful")
    bikes = [a for a in alerts if "bicycle" in a]
    if len(bikes) != 1:
        errs.append(f"expected exactly 1 bicycle proximity alert, got {len(bikes)}: {alerts}")
    if len(alerts) != len(set(alerts)):
        errs.append(f"repeated alert: {alerts}")
    return errs


def check_quiet(run):
    errs = no_errors(run)
    alerts = said(run, r"^Careful")
    if alerts:
        errs.append(f"false proximity alert(s): {alerts}")
    return errs


def check_door(run):
    errs = no_errors(run)
    seen = said(run, r"^Door ")
    near = said(run, r"^Careful, a door")
    if len(seen) != 1:
        errs.append(f"expected the door announced once, got {seen}")
    if len(near) != 1:
        errs.append(f"expected exactly 1 door proximity alert, got {near}")
    return errs


GUIDE_RE = r"^(Found |Straight ahead|A little to your|On your (left|right)|Keep turning|I lost sight|.* should be within reach|.* is right in front of you)"


def check_find(run):
    """'find the bicycle' after turning away: memory -> turn right -> found -> guided -> arrived."""
    errs = no_errors(run)
    first = answer_to(run, "find the bicycle") or ""
    if not first.startswith("Looking for the bicycle"):
        errs.append(f"find not started: {first!r}")
    if "to your right" not in first or "to your left" in first:
        errs.append(f"memory fallback should say turn right (the camera turned left away): {first!r}")
    k = next(i for i, (_, t) in enumerate(run["spoken"]) if t == "[asked: find the bicycle]")
    after = run["spoken"][k + 2:]
    guide = [(f, t) for f, t in after if re.search(GUIDE_RE, t)]
    if not any(t.startswith("Found the bicycle") for _, t in guide):
        errs.append(f"bicycle never announced as found: {after}")
    done = [t for _, t in guide if "right in front of you" in t or "within reach" in t]
    if len(done) != 1:
        errs.append(f"expected exactly one arrival message, got {done}")
    for (f0, a), (f1, b) in zip(guide, guide[1:]):
        if (f1 - f0) / 15.0 < (0.8 if b in done else 1.5) - 0.07:
            errs.append(f"guidance cues too close: {a!r} -> {b!r}")
    alerts = said(run, r"^Careful")
    if len(alerts) != len(set(alerts)):
        errs.append(f"repeated alert: {alerts}")
    return errs


def check_scene_change(run):
    """'I'm leaving' ... dog moved ... 'what changed since I left' ... dog removed ... 'what changed'."""
    errs = no_errors(run)
    mark = answer_to(run, "i'm leaving") or ""
    if not re.search(r"^Okay, I remembered 2 things: .*dog.*bicycle|^Okay, I remembered 2 things: .*bicycle.*dog", mark):
        errs.append(f"snapshot should hold the dog and the bicycle: {mark!r}")
    first = answer_to(run, "what changed since i left") or ""
    if not re.search(r"dog was moved from on your left to on your right", first):
        errs.append(f"the dog's move was not reported: {first!r}")
    second = answer_to(run, "what changed") or ""
    if not re.search(r"dog is no longer where it was", second):
        errs.append(f"the removed dog was not reported: {second!r}")
    for a in (first, second):
        if "New:" in a or "bicycle was moved" in a or "bicycle is no longer" in a:
            errs.append(f"false change reported: {a!r}")
        if re.search(r"(dog|bicycle) \d", a):
            errs.append(f"internal identity number spoken: {a!r}")
    return errs


def check_scene_restart(run):
    """Restart between 'before' and 'after' with only 2 objects (one moved): the
    camera rotation can't be verified, so SENSE must not claim moves - and say why."""
    errs = no_errors(run)
    a = answer_to(run, "what changed since i left") or ""
    if not a.startswith("Since I last ran"):
        errs.append(f"previous run's snapshot not used: {a!r}")
    if "can't tell if anything was moved" not in a:
        errs.append(f"should explain it can't verify moves across the restart: {a!r}")
    if re.search(r"was moved from|no longer where|New:", a):
        errs.append(f"claimed a change it cannot know: {a!r}")
    return errs


def check_people(name, legacy):
    def check(run):
        errs = no_errors(run)
        if not said(run, rf"recognise {name} by face"):
            errs.append(f"enrolment of {name} not confirmed")
        sp = [(f, t) for f, t in run["spoken"]]
        answers = {}
        for k, (f, t) in enumerate(sp):
            if t.startswith("[asked:") and k + 1 < len(sp):
                answers.setdefault(t, []).append((f, sp[k + 1][1]))
        see = answers.get("[asked: what do you see]", [])
        back = [t for f, t in see if 90 <= f < 150]
        others = [t for f, t in see if f >= 150]
        if not back or name.lower() not in back[0].lower():
            errs.append(f"{name} not recognised after leaving and coming back: {back}")
        if any(name.lower() in t.lower() for t in others):
            errs.append(f"someone else was called {name}: {others}")
        names_said = set(re.findall(r"\b([A-Z][a-z]+) (?:is here|in front|on your)", " ".join(t for _, t in sp)))
        stray = names_said - {name, "Careful", "Door", "Currently"}
        if stray:
            errs.append(f"unexpected names spoken: {stray}")
        if legacy:
            ans = answers.get(f"[asked: where is {legacy.lower()}]", [(0, "")])[0][1]
            if "again by face" not in ans:
                errs.append(f"old clothing-colour profile '{legacy}' not handled honestly: {ans!r}")
        return errs
    return check


def run_clip(path, frames, questions, tag, db=None):
    out = os.path.join(tempfile.gettempdir(), f"sense_check_{tag}.json")
    cmd = [sys.executable, os.path.join(ROOT, "tools", "smoke_run.py"), path, str(frames),
           *questions, "--json", out] + (["--db", db] if db else [])
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
    with open(out) as f:
        run = json.load(f)
    run["took"] = time.time() - t0
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", help="folder made by tools/make_test_videos.py")
    ap.add_argument("--name", default="Alex", help="name to teach for person A in people.mp4")
    ap.add_argument("--legacy", default=None,
                    help="a name taught with the OLD clothing method in your sense_memory.db, e.g. Sneha")
    args = ap.parse_args()

    results = []
    print("== Unit and regression tests (tests/)")
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT,
                       capture_output=True, text=True)
    tail = (r.stderr or r.stdout).strip().splitlines()
    summary = next((l for l in reversed(tail) if l.startswith("Ran ")), "?")
    results.append(("unit + regression tests", r.returncode == 0,
                    [] if r.returncode == 0 else tail[-15:], summary + " " + tail[-1]))

    if args.videos:
        n = args.name
        scenarios = [
            ("walk towards objects", "approach_far.mp4", 150, [], check_approach),
            ("stand still near objects", "still_near.mp4", 120, [], check_quiet),
            ("turn the head past objects", "pan_near.mp4", 120, [], check_quiet),
            ("walk towards a door", "approach_door.mp4", 150, [], check_door),
            ("find an item: memory, turn, walk up", "find_item.mp4", 315, ["find the bicycle@150"], check_find),
            ("introduce, leave, come back, strangers", "people.mp4", 308,
             [f"this is {n.lower()}@10", "what do you see@140", "what do you see@200",
              f"where is {n.lower()}@205", "who was here today@265"]
             + ([f"where is {args.legacy.lower()}@300"] if args.legacy else []),
             check_people(n, args.legacy)),
            ("what changed since I left: moved, then removed", "scene_change.mp4", 750,
             ["i'm leaving@120", "what changed since i left@420", "i'm leaving@435", "what changed@735"],
             check_scene_change),
            ("what changed since I left, across a restart", "scene_after.mp4", 180,
             ["what changed since i left@170"], check_scene_restart),
        ]
        for title, clip, frames, qs, check in scenarios:
            path = os.path.join(args.videos, clip)
            if not os.path.exists(path):
                results.append((title, None, [f"skipped: {clip} not found"], ""))
                continue
            print(f"== Scenario: {title} ({clip})")
            db = None
            if clip == "scene_after.mp4":
                # two runs sharing one memory database: 'before' first, then 'after'
                db = os.path.join(tempfile.mkdtemp(), "sense_memory.db")
                run_clip(os.path.join(args.videos, "scene_before.mp4"), 150, [], "scene_before", db=db)
            run = run_clip(path, frames, qs, clip.split(".")[0], db=db)
            errs = check(run)
            spoken = [f"[{f}] {t}" for f, t in run["spoken"][1:]]
            results.append((title, not errs, errs, f"{run['frames']} frames in {run['took']:.0f} s; said: "
                            + (" | ".join(spoken) or "nothing")))

    print("\n================ RESULTS ================")
    failed = 0
    for title, ok, errs, info in results:
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        failed += ok is False
        print(f"{status}  {title}\n      {info}")
        for e in errs:
            print(f"      - {e}")
    print(f"\n{failed} failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
