"""
tools/stress_voice.py - 10,000 SPACE presses through the REAL listener code.

    python tools/stress_voice.py            # 10,000 cycles, ~1 minute
    python tools/stress_voice.py --cycles 500

What is real: voice_input.VoiceInput (thread start, busy flag, stages, guards,
queue hand-off), answers.classify_intent and the name extraction that
"this is Sneha" goes through, and real Python threads with random timing.
What is faked: the microphone and Whisper (no audio hardware here), which
return a random mix of: clear questions, "this is <name>", silence/timeouts,
PortAudio errors, Whisper "no speech" results, junk words and Whisper repeating
its hint prompt. Every RESTART_EVERY cycles the listener is rebuilt from
scratch, like restarting SENSE (new recogniser, new queue, new state).

Checks (any failure is printed and the exit code is 1):
- never more than one listener thread or one open microphone at a time
- every accepted press produces exactly one result; none lost, none doubled
- a press while busy is reported, never silently dropped
- the busy flag is always cleared (SPACE works again after every outcome,
  including errors and restarts)
- an echoed hint prompt or "no speech" never reaches the command code
  (in particular never becomes a FORGET)
- "this is <name>" always arrives as a LEARN with the right name
- no threads are left behind at the end
"""
import argparse
import os
import queue
import random
import re
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import answers  # noqa: E402
import voice_input as vi  # noqa: E402

RESTART_EVERY = 500
NAMES = ["Sneha", "Mummy", "Rahul", "Priya"]
QUESTIONS = ["What do you see?", "Where is my water bottle?", "Find my phone.", "Who was here today?",
             "What changed since I left?", "I'm leaving.", "Speak slower.", "Where is Sneha?",
             "Rename my bottle to medicine bottle.", "Stop looking."]


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.listeners_now = 0
        self.max_listeners = 0
        self.mic_now = 0
        self.max_mic = 0


def build(stats, rng, prompt):
    """A fresh 'SENSE start': new recogniser, queue and VoiceInput."""
    class Mic:
        def __init__(self, device_index=None):
            pass

        def __enter__(self):
            with stats.lock:
                stats.mic_now += 1
                stats.max_mic = max(stats.max_mic, stats.mic_now)
            return self

        def __exit__(self, *a):
            with stats.lock:
                stats.mic_now -= 1

    class SR:
        WaitTimeoutError = type("WaitTimeoutError", (Exception,), {})
        Microphone = Mic

    class Rec:
        energy_threshold = 300
        plan = None                 # set per press: (kind, payload)

        def listen(self, source, timeout=None, phrase_time_limit=None):
            time.sleep(rng.random() * 0.002)
            kind, payload = self.plan
            if kind == "timeout":
                raise SR.WaitTimeoutError("listening timed out")
            if kind == "mic_error":
                raise OSError(payload)
            return payload

        def recognize_whisper(self, audio, model="tiny", initial_prompt="", show_dict=False, language=None):
            time.sleep(rng.random() * 0.002)
            kind, payload = self.plan
            if kind == "whisper_error":
                raise RuntimeError(payload)
            return payload

    q = queue.Queue()

    def start_thread(fn):
        def run():
            with stats.lock:
                stats.listeners_now += 1
                stats.max_listeners = max(stats.max_listeners, stats.listeners_now)
            try:
                fn()
            finally:
                with stats.lock:
                    stats.listeners_now -= 1
        threading.Thread(target=run, daemon=True, name="voice-listener").start()

    rec = Rec()
    v = vi.VoiceInput(SR, rec, q, get_prompt=lambda: prompt, play_cue=lambda: None, stop_speech=lambda: None,
                      log=lambda *a: None, start_thread=start_thread)
    return v, q, rec


def random_plan(rng, prompt):
    r = rng.random()
    seg = lambda nsp, lp: [{"no_speech_prob": nsp, "avg_logprob": lp}]  # noqa: E731
    if r < 0.30:
        name = rng.choice(NAMES)
        text = rng.choice([f" This is {name}.", f"this is {name}", f" This is {name}!", f"My name is {name}."])
        return ("learn", name), ("ok", {"text": text, "segments": seg(0.02, -0.3)})
    if r < 0.55:
        q = rng.choice(QUESTIONS)
        return ("question", q), ("ok", {"text": " " + q, "segments": seg(0.05, -0.4)})
    if r < 0.65:
        return ("timeout", None), ("timeout", None)
    if r < 0.70:
        return ("error", None), ("mic_error", "[Errno -9981] Input overflowed")
    if r < 0.73:
        return ("error", None), ("whisper_error", "Whisper failed")
    if r < 0.82:   # Whisper repeating its hint prompt (whole or part)
        items = vi.prompt_items(prompt)
        k = rng.randint(3, len(items))
        echo = ", ".join(i.title() for i in items[:k]) + "."
        return ("echo", None), ("ok", {"text": " " + echo, "segments": seg(0.3, -0.7)})
    if r < 0.91:
        return ("nospeech", None), ("ok", {"text": " you", "segments": seg(0.92, -1.4)})
    return ("junk", None), ("ok", {"text": rng.choice([" Thank you.", " Bye.", "", " Thanks for watching!"]),
                                   "segments": seg(0.4, -0.8)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    # The prompt SENSE builds after a restart: enrolled names first (refresh_people)
    prompt = ", ".join(NAMES + ["water bottle", "medicine bottle", "phone", "laptop", "rename"]) + "."
    stats = Stats()
    failures = []
    counts = {}
    accepted = ignored_ok = restarts = 0
    v, q, rec = build(stats, rng, prompt)
    threads_at_start = threading.active_count()
    t0 = time.time()

    for i in range(a.cycles):
        if i and i % RESTART_EVERY == 0:          # "quit and start SENSE again"
            v, q, rec = build(stats, rng, prompt)
            restarts += 1
        expect, plan = random_plan(rng, prompt)
        rec.plan = plan
        counts[expect[0]] = counts.get(expect[0], 0) + 1
        if v.is_listening:
            failures.append(f"cycle {i}: busy flag still set before the press")
        if not v.press(rng.choice(["window", "terminal"])):
            failures.append(f"cycle {i}: press refused while idle")
            continue
        accepted += 1
        # impatient extra presses while it is recording/transcribing
        for _ in range(rng.choice([0, 0, 0, 1, 2])):
            if v.is_listening:
                if v.press("window"):
                    failures.append(f"cycle {i}: second listener started while busy")
                else:
                    ignored_ok += 1
        try:
            kind, payload = q.get(timeout=5)
        except queue.Empty:
            failures.append(f"cycle {i}: no result for an accepted press ({expect})")
            continue
        deadline = time.time() + 2
        while v.is_listening and time.time() < deadline:
            time.sleep(0.0002)
        if v.is_listening:
            failures.append(f"cycle {i}: busy flag never cleared ({expect})")
        if not q.empty():
            failures.append(f"cycle {i}: more than one result for one press")
            while not q.empty():
                q.get_nowait()

        if expect[0] in ("learn", "question"):
            if kind != "text":
                failures.append(f"cycle {i}: {expect} became {kind}: {payload}")
                continue
            intent = answers.classify_intent(payload.lower().strip())
            if expect[0] == "learn":
                name = answers.extract_name(payload.lower().strip())
                if intent != "LEARN" or (name or "").strip(" .!?").lower() != expect[1].lower():
                    failures.append(f"cycle {i}: '{payload}' -> {intent}, name {name!r}")
        else:
            if kind != "error":
                failures.append(f"cycle {i}: {expect[0]} reached the command code as {payload!r} "
                                f"(intent {answers.classify_intent(payload.lower())})")
        if len(failures) > 20:
            break

    deadline = time.time() + 3
    while stats.listeners_now and time.time() < deadline:
        time.sleep(0.01)
    if stats.max_listeners > 1:
        failures.append(f"{stats.max_listeners} listener threads at the same time")
    if stats.max_mic > 1:
        failures.append(f"{stats.max_mic} microphones open at the same time")
    if stats.listeners_now or threading.active_count() > threads_at_start:
        failures.append(f"threads left behind: {threading.active_count() - threads_at_start}")

    took = time.time() - t0
    print(f"{accepted} presses in {took:.1f}s, {restarts} restarts, {ignored_ok} extra presses while busy "
          f"(all reported as busy), max listener threads at once: {stats.max_listeners}, "
          f"max open mics: {stats.max_mic}")
    print("outcomes: " + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))
    if failures:
        print(f"FAIL ({len(failures)}):")
        for f in failures[:20]:
            print("  - " + f)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
