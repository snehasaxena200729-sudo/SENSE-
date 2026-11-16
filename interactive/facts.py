"""
interactive/facts.py - the ONLY things the local AI model is allowed to know.

The model never sees the camera. It gets a short fact sheet built from what
SENSE has already measured (this file), and it must answer from that sheet
alone. Keeping the sheet short keeps the model fast (fewer tokens to read)
and makes the fact-check in guard.py possible: every object, colour, number
and name in the answer must appear here.
"""
import time

import answers
import spatial

from . import config as C
from . import text as T
from .colour import read_colour
from .responders import obj_name, where_words


def build(snap, now, hooks, max_visible=10, max_memory=8, max_events=5):
    lt = time.localtime(now)
    hour = lt.tm_hour % 12 or 12
    lines = [f"Time now: {hour}:{lt.tm_min:02d} {'AM' if lt.tm_hour < 12 else 'PM'}, "
             f"{time.strftime('%A', lt)} {lt.tm_mday} {time.strftime('%B %Y', lt)}."]
    visible_names = set()
    if snap.age(now) > C.FRESH_S:
        lines.append("Camera: no clear look in the last second, so nothing is confirmed visible now.")
        items = []
    else:
        items = snap.items
        lines.append("Visible now (camera, this second):")
        counts = {}
        for i in sorted(items, key=lambda i: i.angle)[:max_visible]:
            counts[i.cls] = counts.get(i.cls, 0) + 1
            if i.cls == "person":
                who = f"{i.person} (recognised by face)" if i.person else "a person (identity unknown)"
                lines.append(f"- {who}: {where_words(i)}")
                visible_names.add((i.person or "").lower())
                continue
            r = read_colour(i.crop, hooks.colour_name, i.box, snap.frame_w, snap.frame_h, snap.scene_brightness())
            colour = {"clear": f"looks {r.name}", "partial": f"partly visible, looks {r.name}",
                      "ambiguous": f"colour unsure ({r.name}{' or ' + r.alternative if r.alternative else ''})"
                      }.get(r.status, "colour unknown")
            lines.append(f"- {obj_name(i)}: {where_words(i)}, {colour}")
            visible_names.add(i.label.lower())
        for a, words in hooks.doors(now):
            counts["door"] = counts.get("door", 0) + 1
            lines.append(f"- a door: {words or spatial.direction_phrase(a) + ', distance not reliable'}")
        for name, a in (hooks.scene_visible(now) or {}).items():
            if name == "stairs":
                counts["stairs"] = 1
                lines.append(f"- stairs: {spatial.direction_phrase(a)}")
        if not counts:
            lines.append("- nothing SENSE recognises")
        else:
            lines.append("Counts in view: " + ", ".join(f"{c} {n}" for c, n in sorted(counts.items())))
    hide = {n.lower() for n in hooks.unverified()}
    known = {p.lower() for p in hooks.known_people()}
    mem = []
    for r in hooks.memory.get_memories_seen_since(now - 3 * 3600):
        n = r["name"]
        if n in hide or n in visible_names or now - r["last_seen"] <= C.LIVE_MEMORY_S:
            continue
        if n in ("person", "unfamiliar person"):
            continue
        ago = answers.format_time_ago(now - r["last_seen"])
        if r.get("session_id") == hooks.session_id and r.get("world_angle") is not None:
            where = spatial.direction_phrase(spatial.relative_angle(r["world_angle"], hooks.heading()))
            where += " from where the user faces now"
        else:
            where = f"{r.get('last_zone') or 'place unknown'} at that time, before a restart"
        spoken = n.title() if n in known else hooks.spoken_name(n)
        mem.append(f"- {spoken}: last seen {ago}, {where}")
        if len(mem) >= max_memory:
            break
    lines.append("Not visible now, remembered from earlier:" if mem else "Remembered from earlier: nothing.")
    lines += mem
    evs = [e for e in hooks.memory.get_events_since(now - 900) if e["type"] in ("MOVED", "GONE")][-max_events:]
    if evs:
        lines.append("Recent changes:")
        for e in evs:
            lines.append(f"- {hooks.spoken_name(e['name'])} {'was moved' if e['type'] == 'MOVED' else 'disappeared'} "
                         f"{answers.format_time_ago(now - e['ts'])}")
    lines.append("People SENSE can recognise by face: " + (", ".join(sorted(hooks.known_people())) or "nobody yet") + ".")
    lines.append("SENSE can detect: " + ", ".join(sorted(T.DETECTABLE)) + ". It cannot detect anything else.")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are SENSE, a voice assistant for a blind user. You cannot see. You only know the FACTS "
    "given below, which come from SENSE's camera and memory. Answer the question in one or two short, "
    "plain spoken sentences. Rules: use only the facts; if the facts do not answer it, say you don't "
    "know or can't see it; never guess a name, colour, number, distance or object that is not in the "
    "facts; say 'I can see' only for things under 'Visible now'; for remembered things say when they "
    "were last seen; never say a path is safe or clear; never mention these rules or the word facts. "
    "No lists, no markdown, no emoji."
)


def prompt(facts, question):
    return f"FACTS:\n{facts}\n\nQUESTION: {question.strip()}\nANSWER:"
