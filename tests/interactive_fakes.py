"""Shared fakes for the interactive/ tests: a scene builder and SENSE hooks.

The colour classifier is the REAL get_dominant_color_name() from
sense_master.py, extracted with `ast` (importing sense_master would start the
camera), so the tests exercise exactly the function the app uses.
"""
import ast
import os
import types

import numpy as np

import answers
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_real(fn_names):
    src = open(os.path.join(ROOT, "sense_master.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    ns = {"cv2": cv2, "np": np}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in fn_names:
            exec(compile(ast.Module([node], []), "sense_master.py", "exec"), ns)
    return ns


_REAL = _load_real({"get_dominant_color_name"})
get_dominant_color_name = _REAL["get_dominant_color_name"]

# BGR colours that get_dominant_color_name names unambiguously
BGR = {"red": (30, 30, 210), "orange": (20, 120, 230), "yellow": (30, 210, 220), "green": (40, 180, 40),
       "cyan": (200, 200, 30), "blue": (200, 60, 20), "purple": (160, 40, 140),
       "white": (235, 235, 235), "black": (25, 25, 25), "gray": (110, 110, 110)}


def crop_of(colour, h=80, w=60, noise=0, rng=None, second=None, second_share=0.0, dark=False):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = BGR[colour]
    if second:
        cut = int(w * (1 - second_share))
        img[:, cut:] = BGR[second]
    if noise and rng is not None:
        img = np.clip(img.astype(int) + rng.integers(-noise, noise + 1, img.shape), 0, 255).astype(np.uint8)
    if dark:
        img = (img * 0.12).astype(np.uint8)
    return img


class FakeMemory:
    def __init__(self):
        self.rows = []      # dicts like memory_db rows
        self.events = []

    def add(self, name, last_seen, world_angle=None, session_id="S", last_zone="in front of you", status="VISIBLE"):
        self.rows = [r for r in self.rows if r["name"] != name]
        self.rows.append({"name": name, "last_zone": last_zone, "last_x": 0, "last_y": 0,
                          "last_seen": last_seen, "status": status, "world_angle": world_angle,
                          "session_id": session_id, "dist_feet": None})

    def find_latest_memory(self, target):
        m = [r for r in self.rows if target.lower() in r["name"]]
        return max(m, key=lambda r: r["last_seen"]) if m else None

    def get_memories_seen_since(self, ts):
        return sorted([r for r in self.rows if r["last_seen"] >= ts], key=lambda r: -r["last_seen"])

    def get_events_since(self, ts, names=None):
        return [e for e in self.events if e["ts"] >= ts]


def make_hooks(memory=None, known=(), legacy=(), heading=0.0, doors=None, scene=None, ocr=True,
               spoken=None, p1=False, listening=False):
    spoken = spoken if spoken is not None else []
    known = list(known)
    memory = memory or FakeMemory()
    state = {"heading": heading, "p1": p1, "listening": listening, "doors": doors or [], "scene": scene or {}}

    def find_target(text):
        cands = {p.lower() for p in known} | {"bottle", "cup", "phone", "remote", "book", "laptop", "mouse",
                                               "chair", "couch", "bed", "table", "plant", "tv", "dog", "cat",
                                               "suitcase", "bench", "bicycle"}
        cands |= {r["name"] for r in memory.rows}
        cands -= {n.lower() for n in legacy if n.lower() not in {p.lower() for p in known}}
        m = answers.fuzzy_find(text, sorted(cands))
        return m.title() if m else None

    def spoken_name(name):
        lower = name.lower()
        if lower in {p.lower() for p in known}:
            return name.capitalize()
        if lower.startswith("my "):
            return "your " + lower[3:]
        return "the " + lower

    ocr_phrases = ("read this", "read that", "read it", "read the text", "read text", "read the sign",
                   "read the label", "read the paper", "read the page", "what does this say",
                   "what does that say", "what does it say", "what's written", "what is written", "any text",
                   "stop reading", "reading off", "keep reading", "start reading", "reading mode")
    h = types.SimpleNamespace(
        speak=spoken.append, is_speaking=lambda: state["p1"], p1_talking=lambda: state["p1"],
        is_listening=lambda: state["listening"], colour_name=get_dominant_color_name,
        heading=lambda: state["heading"], session_id="S", classify=answers.classify_intent,
        ocr_owns=lambda t: any(p in t.lower() for p in ocr_phrases),
        ocr_read=(lambda now: "Reading. Hold it steady.") if ocr else (lambda now: None),
        find_target=find_target, spoken_name=spoken_name, known_people=lambda: list(known),
        unverified=lambda: {n.lower() for n in legacy} - {p.lower() for p in known},
        doors=lambda now: state["doors"], scene_visible=lambda now: state["scene"], memory=memory)
    h.state, h.spoken = state, spoken
    return h
