"""
manager.py - PersonalItems, the only class sense_master.py talks to.

    label(tid, category, sig, world_angle, now, stable)  per confirmed track
    end_frame(now, stable)                                once per frame
    pick_teach_target(candidates)                         "this is my cup": which cup?
    teach(name, category, sig, tid, world_angle, now, other_sigs)
    query(text, now)          "where is my cup?" -> ("target", name) / ("say", text) / None
    note(name, now, heading)  extra honest sentence for the answer, or ""
    forget(name) / rename(old, new)

Every public method is wrapped: an error is printed once and SENSE carries on;
sense_master falls back to its old matching when `enabled` is False.
"""

import functools

from . import phrasing
from .config import PersonalConfig
from .matcher import Item, PersonalMatcher
from .store import PersonalStore


def _safe(default=None):
    def wrap(fn):
        @functools.wraps(fn)
        def inner(self, *a, **kw):
            if not self.enabled:
                return default
            try:
                return fn(self, *a, **kw)
            except Exception as e:      # never let this feature break SENSE
                if fn.__name__ not in self._errors:
                    self._errors.add(fn.__name__)
                    print(f"🔐 Personal items {fn.__name__} error (ignored): {e!r}")
                return default
        return inner
    return wrap


class PersonalItems:
    def __init__(self, conn, session_id, known_objects=None, known_categories=(), cfg=None):
        self.cfg = cfg or PersonalConfig()
        self.enabled = True
        self._errors = set()
        self.known_categories = set(known_categories) | {"bag"}
        self.store = PersonalStore(conn)
        self.matcher = PersonalMatcher(session_id, self.cfg)
        self._last_save = {}
        for item in self.store.load():
            # custom_object_profiles stays the list of what is personal (forget
            # and rename keep it in step); rows missing there are not loaded
            if known_objects is None or item.name in known_objects:
                self.matcher.add_item(item)
        known_objects = known_objects or {}
        # one-time migration: items taught before this package existed
        for name, d in known_objects.items():
            if name not in self.matcher.items and d.get("sig") is not None:
                item = Item(name, d["category"], d["sig"].copy(), f"{d['category']}-legacy-{len(self.matcher.items)}", 0.0)
                self.matcher.add_item(item)
                self.store.save(item)

    # -------------------------------------------------------------- seeing
    def has_category(self, category):
        return self.enabled and category in self.matcher.categories()

    @_safe()
    def label(self, tid, category, sig, world_angle, now, stable):
        return self.matcher.label(tid, category, sig, world_angle, now, stable)

    @_safe()
    def end_frame(self, now, stable):
        self.matcher.end_frame(now, stable)
        for name in list(self.matcher.dirty):
            item = self.matcher.items.get(name)
            if item is None:
                self.matcher.dirty.discard(name)
                continue
            if now - self._last_save.get(name, -1e9) >= self.cfg.save_every_s:
                self.store.save(item)
                self._last_save[name] = now
                self.matcher.dirty.discard(name)

    def flush(self):
        """Write everything (called on exit)."""
        if not self.enabled:
            return
        try:
            for name in list(self.matcher.dirty):
                if name in self.matcher.items:
                    self.store.save(self.matcher.items[name])
            self.matcher.dirty.clear()
        except Exception as e:
            print(f"🔐 Personal items flush error (ignored): {e!r}")

    # ------------------------------------------------------------ teaching
    def pick_teach_target(self, candidates):
        """candidates: [(tid, area)] of the spoken kind. Returns (tid, ok).

        ok is False when two are almost the same size: we can't know which one
        the user means, so we ask instead of guessing."""
        if not candidates:
            return None, False
        ranked = sorted(candidates, key=lambda c: -c[1])
        if len(ranked) >= 2 and ranked[1][1] >= self.cfg.teach_runner_up_ratio * ranked[0][1]:
            return ranked[0][0], False
        return ranked[0][0], True

    @_safe()
    def teach(self, name, category, sig, tid, world_angle, now, other_sigs=()):
        item = self.matcher.teach(name, category, sig, tid, world_angle, now, other_sigs)
        self.store.save(item)
        self._last_save[name] = now
        self.matcher.dirty.discard(name)
        print(f"🔐 '{name}' is instance {item.instance_id} ({len(item.lookalikes)} look-alikes noted)")
        return item

    @_safe()
    def forget(self, name):
        self.matcher.forget(name.lower())
        self.store.forget(name)

    @_safe()
    def rename(self, old, new):
        self.matcher.rename(old.lower(), new.lower())
        self.store.rename(old, new)
        if new.lower() in self.matcher.items:
            self.store.save(self.matcher.items[new.lower()])

    # ----------------------------------------------------------- answering
    @_safe()
    def query(self, text, now=None):
        if not self.matcher.items and "my " not in text.lower():
            return None
        return phrasing.resolve(text, self.matcher.items, self.known_categories, self.cfg.synonyms)

    @_safe(default="")
    def note(self, name, now, heading):
        return phrasing.note(self.matcher.status(name.lower(), now), now, heading)


class _Disabled:
    enabled = False

    def has_category(self, *a, **kw):
        return False

    def label(self, *a, **kw):
        return None

    end_frame = teach = forget = rename = query = flush = label

    def pick_teach_target(self, candidates):
        ranked = sorted(candidates, key=lambda c: -c[1])
        return (ranked[0][0], True) if ranked else (None, False)

    def note(self, *a, **kw):
        return ""


def create(session_id, known_objects=None, known_categories=(), conn=None):
    """Make PersonalItems on memory_db's connection; never raises."""
    try:
        if conn is None:
            import memory_db
            conn = memory_db._get_conn()
        p = PersonalItems(conn, session_id, known_objects, known_categories)
        print(f"🔐 Personal items: each one is a single object, never a whole kind "
              f"({len(p.matcher.items)} known).")
        return p
    except Exception as e:
        print(f"🔐 Personal items off, using the old colour match ({e!r})")
        return _Disabled()
