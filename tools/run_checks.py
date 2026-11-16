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


def check_door_distance(run):
    """Door distance: one first sighting with a number or an honest 'cannot estimate',
    updates only getting closer while walking in, and a 'where is the door' answer."""
    errs = no_errors(run)
    first = said(run, r"^Door detected")
    if len(first) != 1:
        errs.append(f"expected one 'Door detected ...', got {first}")
    elif not re.search(r"approximately \d+ feet|cannot estimate its distance reliably|far ahead", first[0]):
        errs.append(f"first sighting without a distance or an honest 'cannot estimate': {first}")
    nums = [int(m.group(1)) for t in said(run, r"^(Door detected|The door is)")
            for m in [re.search(r"approximately (\d+) feet", t)] if m]
    if nums != sorted(nums, reverse=True):
        errs.append(f"distances while walking towards the door must not grow: {nums}")
    ans = answer_to(run, "where is the door")
    if not ans or not re.search(r"door", ans, re.I) or re.search(r"haven't seen", ans):
        errs.append(f"bad answer to 'where is the door': {ans!r}")
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


DESTRUCTIVE_RE = r"forgotten|is now called|memorized|please look at the camera"


def check_echo(run, name):
    """Whisper repeating its own hint prompt must not change memory (regression)."""
    sp = run["spoken"]
    k = next((i for i, (_, t) in enumerate(sp) if t.startswith("[asked:") and "water bottle" in t), None)
    if k is None:
        return ["the Whisper echo question was not delivered"]
    reply = sp[k + 1][1] if k + 1 < len(sp) else ""
    errs = [f"echo was obeyed as a command: {reply!r}"] if re.search(DESTRUCTIVE_RE, reply, re.I) else []
    who = answer_to(run, "who was here today") or ""
    if name.lower() not in who.lower():
        errs.append(f"{name} missing after the echo: {who!r}")
    return errs


def check_sway_naming(name):
    def check(run):
        errs = no_errors(run)
        first = answer_to(run, f"this is {name.lower()}") or ""
        if "look at the camera" not in first:
            errs.append(f"naming failed while the image moves: {first!r}")
        if not said(run, rf"recognise {name} by face"):
            errs.append(f"enrolment of {name} not confirmed")
        return errs
    return check


def check_events(a, b):
    """events.mp4: a disappears in view, camera turns away and back, a is back elsewhere."""
    def check(run):
        errs = no_errors(run)
        where_gone = answer_to(run, f"where is the {a}") or ""
        if "disappeared" not in where_gone or "probably still there" in where_gone or where_gone.startswith("I see"):
            errs.append(f"gone {a} not reported honestly: {where_gone!r}")
        find = answer_to(run, f"find the {a}") or ""
        if "disappeared" not in find:
            errs.append(f"find mode hides that the {a} disappeared: {find!r}")
        sp = run["spoken"]
        changes = [sp[i + 1][1] for i, (_, t) in enumerate(sp[:-1]) if t == "[asked: what changed]"]
        if len(changes) != 2:
            return errs + [f"expected 2 'what changed' answers, got {changes}"]
        if "disappeared" not in changes[0] or a not in changes[0]:
            errs.append(f"first 'what changed' should say the {a} disappeared: {changes[0]!r}")
        if "moved" not in changes[1] or a not in changes[1]:
            errs.append(f"second 'what changed' should say the {a} was moved: {changes[1]!r}")
        if any(b in c for c in changes):
            errs.append(f"turning the camera away made the {b} count as a change: {changes}")
        before = answer_to(run, f"where was the {a} before") or ""
        if "on your left" not in before or "on your right" not in before:
            errs.append(f"previous place of the {a} wrong: {before!r}")
        return errs
    return check


def run_clip(path, frames, questions, tag):
    out = os.path.join(tempfile.gettempdir(), f"sense_check_{tag}.json")
    cmd = [sys.executable, os.path.join(ROOT, "tools", "smoke_run.py"), path, str(frames),
           *questions, "--json", out]
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
    ap.add_argument("--event-names", nargs=2, default=["dog", "bicycle"], metavar=("A", "B"),
                    help="class names of the two objects in events.mp4 (default: dog bicycle)")
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

    print("== OCR tests (ocr/tests/)")   # optional text reading module, tested on its own
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "ocr/tests", "-t", "."],
                       cwd=ROOT, capture_output=True, text=True)
    tail = (r.stderr or r.stdout).strip().splitlines()
    summary = next((l for l in reversed(tail) if l.startswith("Ran ")), "?")
    results.append(("OCR tests", r.returncode == 0, [] if r.returncode == 0 else tail[-15:], summary + " " + tail[-1]))

    print("== Whereabouts tests (whereabouts/tests/)")   # "where did I leave it?" module
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "whereabouts/tests", "-t", "."],
                       cwd=ROOT, capture_output=True, text=True)
    tail = (r.stderr or r.stdout).strip().splitlines()
    summary = next((l for l in reversed(tail) if l.startswith("Ran ")), "?")
    results.append(("whereabouts tests", r.returncode == 0, [] if r.returncode == 0 else tail[-15:],
                    summary + " " + tail[-1]))

    print("== Interactive questions: 10,000-case stress test (tools/stress_interactive.py)")
    r = subprocess.run([sys.executable, os.path.join("tools", "stress_interactive.py")], cwd=ROOT,
                       capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()
    results.append(("interactive questions stress test", r.returncode == 0,
                    [] if r.returncode == 0 else tail[-15:], (tail[-1] if tail else "?") + " (10,000 cases)"))

    print("== Personal items: 10,000-case stress test (tools/stress_personal.py)")   # "my cup" = one cup
    r = subprocess.run([sys.executable, os.path.join("tools", "stress_personal.py")], cwd=ROOT,
                       capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()
    results.append(("personal items stress test", r.returncode == 0,
                    [] if r.returncode == 0 else tail[-15:], tail[-1] if tail else "?"))

    if args.videos:
        n = args.name
        ea, eb = args.event_names
        scenarios = [
            ("walk towards objects", "approach_far.mp4", 150, [], check_approach),
            ("stand still near objects", "still_near.mp4", 120, [], check_quiet),
            ("turn the head past objects", "pan_near.mp4", 120, [], check_quiet),
            ("walk towards a door", "approach_door.mp4", 150, [], check_door),
            ("door distance: announce, update, answer", "approach_door.mp4", 150,
             ["where is the door@100"], check_door_distance),
            ("find an item: memory, turn, walk up", "find_item.mp4", 315, ["find the bicycle@150"], check_find),
            ("introduce, leave, come back, strangers", "people.mp4", 308,
             [f"this is {n.lower()}@10", "what do you see@140", "what do you see@200",
              f"where is {n.lower()}@205", "who was here today@265"]
             + ([f"where is {args.legacy.lower()}@300"] if args.legacy else []),
             check_people(n, args.legacy)),
            ("Whisper echoes its hint prompt (must not forget anything)", "people.mp4", 280,
             [f"this is {n.lower()}@10", "<whisper echo>@200", "who was here today@240"],
             lambda run: no_errors(run) + check_echo(run, n)),
            ("introduce someone while the image moves", "sway_person.mp4", 240,
             [f"this is {n.lower()}@40"], check_sway_naming(n)),
            ("object disappears, camera turns away and back, object moved", "events.mp4", 770,
             [f"where is the {ea}@305", f"find the {ea}@400", "stop looking@405", "what changed@460",
              "what changed@740", f"where was the {ea} before@750"], check_events(ea, eb)),
        ]
        for title, clip, frames, qs, check in scenarios:
            path = os.path.join(args.videos, clip)
            if not os.path.exists(path):
                results.append((title, None, [f"skipped: {clip} not found"], ""))
                continue
            print(f"== Scenario: {title} ({clip})")
            run = run_clip(path, frames, qs, clip.split(".")[0])
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
