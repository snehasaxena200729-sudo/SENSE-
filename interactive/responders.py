"""
interactive/responders.py - one small function per question type.

Each one answers ONLY from: the last YOLO frame (snapshot), SENSE's memory
database, the door model, or the clock. Every sentence says which of these
it comes from - "I can see" (now), "I saw ... minutes ago" (memory), or
"I'm not sure" (with the reason). Nothing is guessed.
"""
import time

import answers
import spatial

from . import config as C
from . import text as T
from .colour import read_colour, voted_colour

SIDE_WORDS = {"ahead": "in front of you", "left": "on your left", "right": "on your right",
              "behind": "behind you"}


HANDHELD = {"phone", "remote", "cup", "bottle", "book", "laptop", "mouse"}


class Rewrite(str):
    """Not an answer: hand this text to the existing SENSE question code."""


# ---------------------------------------------------------------- helpers
def stale_sentence(snap, now):
    return "I haven't had a clear look in the last second. Hold the camera still and ask again."


def obj_name(item):
    """Spoken name for one visible thing; people only by verified face."""
    if item.cls == "person":
        return item.person or "a person"
    if item.label.lower().startswith("my "):
        return "your " + item.label.lower()[3:]
    return "the " + item.cls


def a_name(item):
    if item.cls == "person":
        return item.person or "a person"
    if item.label.lower().startswith("my "):
        return "your " + item.label.lower()[3:]
    return f"{T.article(item.cls)} {item.cls}"


def where_words(item, with_distance=True):
    d = item.direction
    if with_distance and item.dist_phrase and item.dist_phrase != "distance unclear":
        return f"{d}, {item.dist_phrase}"
    return d


def visible(snap, now):
    return snap.items if snap.age(now) <= C.FRESH_S else None


def side_ok(item, side):
    return side is None or item.side == side


def match_items(q, snap, now, ctx):
    """The visible things the question is about, plus the subject used."""
    items = visible(snap, now)
    if items is None:
        return None, None
    subj = ctx.get(now) if q.pronoun else None
    if q.name:
        n = q.name.lower()
        out = [i for i in items if i.label.lower() == n or (i.person or "").lower() == n]
        if not out and T.CLASS_WORDS.get(n.split()[-1]):
            # 'white bottle' named in memory, maybe labelled differently now
            out = [i for i in items if i.cls == T.CLASS_WORDS[n.split()[-1]] and n in i.label.lower()]
    elif q.cls:
        out = [i for i in items if i.cls == q.cls]
    elif subj is not None:
        if subj.tid is not None and any(i.tid == subj.tid for i in items):
            out = [i for i in items if i.tid == subj.tid]
        elif subj.name:
            out = [i for i in items if i.label.lower() == subj.name or (i.person or "").lower() == subj.name]
        else:
            out = [i for i in items if i.cls == subj.cls]
    else:
        out = None    # nothing named: caller picks the thing in front of the camera
    if out is not None and q.side:
        out = [i for i in out if side_ok(i, q.side)]
    return out, subj


def centre_object(items, people=False):
    """'this' = the object nearest the middle of the view (the biggest wins a tie)."""
    pool = [i for i in items if people or i.cls != "person"]
    if not pool:
        return None
    def area(i):
        return (i.box[2] - i.box[0]) * (i.box[3] - i.box[1])
    return min(pool, key=lambda i: (round(abs(i.angle) / 8.0), -area(i)))


def subject_label(q, subj):
    if q.name:
        return q.name.lower()
    if q.cls:
        return q.cls
    if subj is not None:
        return subj.name or subj.cls
    return None


def the(label):
    if not label:
        return "it"
    if label.startswith("my "):
        return "your " + label[3:]
    if label in T.DETECTABLE or label.split()[-1] in T.CLASS_WORDS:
        return "the " + label
    return label.title()


# ---------------------------------------------------------------- clock
def time_answer(now):
    lt = time.localtime(now)
    hour = lt.tm_hour % 12 or 12
    return f"It is {hour}:{lt.tm_min:02d} {'AM' if lt.tm_hour < 12 else 'PM'}."


def date_answer(now):
    lt = time.localtime(now)
    return f"Today is {time.strftime('%A', lt)}, {lt.tm_mday} {time.strftime('%B %Y', lt)}."


# ---------------------------------------------------------------- colour
def colour_sentence(item, reading, asked=None):
    who = obj_name(item)
    Who = who[0].upper() + who[1:]
    voted = voted_colour(item.label)
    if reading.status in ("too_small", "none"):
        return f"{Who} is too small in the picture for me to tell its colour. Bring it closer."
    if reading.status == "too_dark":
        return f"It is too dark for me to tell the colour of {who}."
    c, alt = reading.name, reading.alternative
    if asked:
        if reading.status == "clear" and c == asked:
            return f"Yes, {who} looks {c}."
        if reading.status == "clear":
            return f"No, {who} looks {c}, not {asked}."
        if asked in (c, alt):
            return f"Maybe. {Who} looks {c}{' or ' + alt if alt else ''}, so I'm not sure."
        if reading.status == "partial":
            return f"I can only see part of {who}. That part looks {c}, not {asked}."
    if reading.status == "partial":
        s = f"I can only see part of {who}. That part looks {c}."
    elif reading.status == "ambiguous":
        s = (f"{Who} looks {c} or {alt}. I'm not sure." if alt else
             f"I'm not sure about the colour of {who}. It might be {c}.")
    else:
        s = f"{Who} looks {c}."
    if voted and voted != c and alt != voted and reading.status == "clear":
        s += f" Earlier it looked {voted} to me, so the light may be changing it."
    return s


def colour_answer(q, snap, now, hooks, ctx):
    if q.undetectable:
        return f"I can't detect {q.undetectable}, so I can't tell the colour."
    subj = ctx.get(now) if q.pronoun else None
    cls = q.cls or (subj.cls if subj else None)
    if cls == "person" or (subj and subj.name and subj.name in [p.lower() for p in hooks.known_people()]):
        return "I only tell the colour of objects, not of people."
    if cls in T.SCENE_CLASSES:
        return f"I can find {T.plural(cls, 2)}, but I can't judge their colour yet."
    items, subj = match_items(q, snap, now, ctx)
    if items is None and visible(snap, now) is None:
        return stale_sentence(snap, now)
    if items is None:                                  # "what colour is this?"
        item = centre_object(visible(snap, now))
        if item is None:
            if q.pronoun and not ctx.get(now):
                return "Which object do you mean? Point the camera at it and ask again."
            return "I don't see an object I recognise in front of the camera."
        items = [item]
    if not items:
        return colour_from_memory(q, subj, now, hooks)
    readings = [(i, read_colour(i.crop, hooks.colour_name, i.box, snap.frame_w, snap.frame_h, snap.scene_brightness()))
                for i in items[:3]]
    if len(items) == 1:
        item, r = readings[0]
        ctx.set(now, cls=item.cls, name=None if item.label.lower() == item.cls else item.label, tid=item.tid)
        return colour_sentence(item, r, q.colour)
    ctx.set(now, cls=items[0].cls)
    parts = []
    for item, r in readings:
        if r.status in ("clear", "partial") and r.name:
            parts.append(f"one looks {r.name}{' (only partly visible)' if r.status == 'partial' else ''}, "
                         f"{item.direction}")
        elif r.status == "ambiguous" and r.name:
            parts.append(f"one looks {r.name}{' or ' + r.alternative if r.alternative else ''}, {item.direction}")
        else:
            parts.append(f"one I can't judge, {item.direction}")
    head = f"I can see {T.number_word(len(items))} {T.plural(items[0].cls, len(items))}"
    more = f", and {len(items) - 3} more" if len(items) > 3 else ""
    return f"{head}: " + "; ".join(parts) + more + ". Which one do you mean?"


def colour_from_memory(q, subj, now, hooks):
    label = subject_label(q, subj)
    if not label:
        return "I don't see that right now."
    rec = hooks.memory.find_latest_memory(label)
    where = f" {SIDE_WORDS[q.side]}" if q.side else ""
    if rec is None:
        return f"I can't see {the(label)}{where} right now, and I don't remember seeing it."
    c = voted_colour(rec["name"])
    ago = answers.format_time_ago(now - rec["last_seen"])
    if c:
        return f"I can't see {the(label)}{where} right now. When I saw it {ago}, it looked {c}."
    return f"I can't see {the(label)}{where} right now, so I can't tell its colour. I last saw it {ago}."


# ---------------------------------------------------------------- count
def _door_angles(hooks, now, cls):
    if cls == "door":
        return [a for a, _ in hooks.doors(now)]
    vis = hooks.scene_visible(now) or {}
    return [vis[cls]] if cls in vis else []


def count_answer(q, snap, now, hooks, ctx):
    if q.undetectable:
        return f"I can't detect {q.undetectable}, so I can't count them."
    cls = q.cls
    if cls is None and q.pronoun:
        s = ctx.get(now)
        cls = s.cls if s else None
    side = q.side if q.side in ("left", "right") else None   # "in front of me" = what the camera sees
    where = f" {SIDE_WORDS[side]}" if side else ""
    if cls in T.SCENE_CLASSES:
        angles = [a for a in _door_angles(hooks, now, cls)
                  if side is None or {"in front of you": "ahead", "on your left": "left",
                                      "on your right": "right"}.get(spatial.direction_phrase(a)) == side]
        n = len(angles)
        ctx.set(now, cls=cls)
        if cls == "stairs":
            return f"I can see stairs{where}." if n else f"I don't see any stairs{where} right now."
        return (f"I can see {T.number_word(n)} {T.plural('door', n)}{where}." if n
                else f"I don't see a door{where} right now.")
    items = visible(snap, now)
    if items is None:
        return stale_sentence(snap, now)
    items = [i for i in items if side_ok(i, side)]
    if cls is not None:
        ctx.set(now, cls=cls)
        sel = [i for i in items if i.cls == cls]
        n = len(sel)
        if n == 0:
            if cls == "person":
                return f"I don't see anyone{where} right now."
            return f"I don't see any {T.plural(cls, 2)}{where} right now."
        if cls == "person":
            named = sorted({i.person for i in sel if i.person})
            unknown = n - sum(1 for i in sel if i.person)
            head = f"I can see {T.number_word(n)} {T.plural('person', n)}{where}"
            if named:
                bits = [T.join_and(named)]
                if unknown:
                    bits.append(f"{T.number_word(unknown)} {'person' if unknown == 1 else 'people'} "
                                f"I don't recognise")
                return f"{head}: " + " and ".join(bits) + "."
            return head + "." if n > 1 else f"I can see one person{where}, {sel[0].direction}."
        if n == 1:
            return f"I can see one {cls}{where}, {where_words(sel[0])}."
        by_dir = {}
        for i in sel:
            by_dir[i.direction] = by_dir.get(i.direction, 0) + 1
        spread = T.join_and([f"{T.number_word(k)} {v}" for v, k in by_dir.items()])
        return f"I can see {T.number_word(n)} {T.plural(cls, n)}{where}: {spread}."
    # "how many things / objects can you see?"
    counts = {}
    for i in items:
        counts[i.cls] = counts.get(i.cls, 0) + 1
    for d in _door_angles(hooks, now, "door"):
        if side is None:
            counts["door"] = counts.get("door", 0) + 1
    total = sum(counts.values())
    if total == 0:
        return f"I don't see anything I recognise{where} right now."
    parts = [f"{T.number_word(n) if n > 1 else T.article(c)} {T.plural(c, n)}"
             for c, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    return f"I can see {T.number_word(total)} {'thing' if total == 1 else 'things'}{where}: {T.join_and(parts)}."


# ---------------------------------------------------------------- side
def side_answer(q, snap, now, hooks, ctx):
    side = q.side or "ahead"
    words = SIDE_WORDS[side]
    items = visible(snap, now)
    if items is None:
        if q.safety:
            return ("I can't see clearly right now, so I can't check the way ahead. "
                    "Please stop and hold the camera still for a moment.")
        return stale_sentence(snap, now)
    sel = [i for i in items if i.side == side]
    door_bits = []
    for name, angle in (hooks.scene_visible(now) or {}).items():
        if name in ("door", "stairs") and {"in front of you": "ahead", "on your left": "left",
                                           "on your right": "right"}.get(spatial.direction_phrase(angle)) == side:
            if name == "door":
                dw = [w for a, w in hooks.doors(now) if abs(a - angle) < 15 and w]
                door_bits.append("a door" + (f", {dw[0].replace(' ' + words, '')}" if dw else ""))
            else:
                door_bits.append("stairs")
    if q.cls:
        ctx.set(now, cls=q.cls)
        if q.cls in T.SCENE_CLASSES:
            hit = [b for b in door_bits if q.cls in b]
            return (f"Yes, there is {hit[0]} {words}." if hit
                    else f"No, I don't see {'stairs' if q.cls == 'stairs' else 'a door'} {words}.")
        hit = [i for i in sel if i.cls == q.cls]
        if hit:
            ctx.set(now, cls=q.cls, tid=hit[0].tid if len(hit) == 1 else None)
            if len(hit) == 1:
                return f"Yes, there is {a_name(hit[0])} {words}" + (
                    f", {hit[0].dist_phrase}." if hit[0].dist_phrase != "distance unclear" else ".")
            return f"Yes, there are {T.number_word(len(hit))} {T.plural(q.cls, len(hit))} {words}."
        elsewhere = [i for i in items if i.cls == q.cls]
        if q.cls == "person":
            base = f"No, I don't see anyone {words}."
        else:
            base = f"No, I don't see {T.article(q.cls)} {q.cls} {words}."
        if elsewhere:
            base += f" I can see {a_name(elsewhere[0])} {elsewhere[0].direction}."
        return base
    parts = [f"{a_name(i)}, {i.dist_phrase}" if i.dist_phrase != "distance unclear" else a_name(i)
             for i in sorted(sel, key=lambda i: (i.dist_m is None, i.dist_m or 0))[:C.MAX_LIST]]
    parts += door_bits
    if len(sel) + len(door_bits) > C.MAX_LIST:
        parts.append(f"{len(sel) + len(door_bits) - C.MAX_LIST} more")
    if q.safety:
        if not parts:
            return ("I don't see any obstacle I recognise in front of you, but I can't detect "
                    "everything, so please go carefully.")
        return (f"Careful. In front of you I can see {T.join_and(parts)}. "
                f"I can't promise the way is clear.")
    if not parts:
        return f"I don't see anything I recognise {words} right now."
    if len(sel) == 1 and not door_bits:
        ctx.set(now, cls=sel[0].cls, tid=sel[0].tid)
    return f"{words[0].upper() + words[1:]}, I can see {T.join_and(parts)}."


# ---------------------------------------------------------------- describe
def describe_answer(q, snap, now, hooks, ctx):
    items, subj = match_items(q, snap, now, ctx)
    if items is None and visible(snap, now) is None:
        return stale_sentence(snap, now)
    if items is None:
        pool = visible(snap, now)
        if q.handheld:
            pool = [i for i in pool if i.cls in HANDHELD]
        item = centre_object(pool, people=not q.handheld)
        if item is None:
            return ("I don't see anything I recognise in your hand. Hold it up in front of the camera."
                    if q.handheld else "I don't see an object I recognise in front of the camera.")
        items = [item]
    if not items:
        label = subject_label(q, subj)
        if label:
            return Rewrite(f"where is {the(label) if not label.startswith('my ') else label}")
        return "I don't see that right now."
    item = items[0]
    ctx.set(now, cls=item.cls, name=None if item.label.lower() == item.cls else item.label, tid=item.tid)
    first = a_name(item)
    s = f"{first[0].upper() + first[1:]}, {where_words(item)}."
    if item.cls == "person":
        if not item.person:
            s += " I don't recognise them."
        return s + (f" I can see {len(items) - 1} more." if len(items) > 1 else "")
    r = read_colour(item.crop, hooks.colour_name, item.box, snap.frame_w, snap.frame_h, snap.scene_brightness())
    if r.status == "clear":
        s += f" It looks {r.name}."
    elif r.status == "partial":
        s += f" I can only see part of it; that part looks {r.name}."
    elif r.status == "ambiguous" and r.name:
        s += f" It looks {r.name}{' or ' + r.alternative if r.alternative else ''}."
    else:
        s += " I can't tell its colour."
    if len(items) > 1:
        s += f" There {'is' if len(items) == 2 else 'are'} {len(items) - 1} more like it."
    return s


# ---------------------------------------------------------------- earlier / behind
def _direction_of(rel):
    a = abs(rel)
    if a <= C.DIRECTION_HALF_WIDTH["ahead"]:
        return "ahead"
    if a > 135:
        return "behind"
    return "right" if rel > 0 else "left"


def past_answer(q, snap, now, hooks, ctx):
    side = q.side or "ahead"
    words = SIDE_WORDS[side]
    heading = hooks.heading()
    hide = {n.lower() for n in hooks.unverified()}
    known = {p.lower() for p in hooks.known_people()}
    rows = []
    for r in hooks.memory.get_memories_seen_since(now - C.PAST_WINDOW_S):
        if r.get("session_id") != hooks.session_id or r.get("world_angle") is None:
            continue                     # angles from an earlier run are meaningless now
        if now - r["last_seen"] <= C.LIVE_MEMORY_S and side != "behind":
            continue                     # visible now: not "previously"
        name = r["name"]
        if name in hide or name in ("person", "unfamiliar person"):
            continue
        if q.cls and T.CLASS_WORDS.get(name.split()[-1]) != q.cls and name not in known:
            continue
        if q.cls == "person" and name not in known:
            continue
        if _direction_of(spatial.relative_angle(r["world_angle"], heading)) == side:
            rows.append(r)
    rows.sort(key=lambda r: -r["last_seen"])
    lead = "I can't see behind you. " if side == "behind" else ""
    if not rows:
        what = f"{T.plural(q.cls, 2)} " if q.cls and q.cls != "person" else ("anyone " if q.cls == "person" else "anything ")
        return f"{lead}I don't remember {what}{words} from the last 30 minutes."
    parts = []
    for r in rows[:C.MAX_LIST]:
        n = r["name"]
        spoken = n.title() if n in known else hooks.spoken_name(n)
        parts.append(f"{spoken} {answers.format_time_ago(now - r['last_seen'])}")
    more = f", and {len(rows) - C.MAX_LIST} more" if len(rows) > C.MAX_LIST else ""
    if len(rows) == 1:
        ctx.set(now, name=rows[0]["name"])
    return f"{lead}Earlier, {words} from where you face now, I saw {T.join_and(parts)}{more}."


def read_answer(q, snap, now, hooks, ctx):
    reply = hooks.ocr_read(now)
    return reply or "Text reading is off on this computer, so I can't read labels."
