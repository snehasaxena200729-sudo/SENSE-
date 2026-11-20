"""
manager.py - HomeMap, the only class sense_master.py talks to.

    handle(text, now, known_names) -> sentence to speak, or None
        None = not a room command; SENSE's existing voice code carries on
        with the sentence exactly as before.
    observe(frame, boxes, names, heading, camera_stable, now, can_speak) -> sentence or None
        Once per frame (Phase 1). Saves views while you teach a room, and on
        every 10th steady frame checks which room the camera sees. Returns a
        teaching message ("Done. I've learned the kitchen") only when SENSE
        may speak (not listening, not already talking).

Every public method is wrapped: an error is printed once and the method
returns None, so a bug here can never stop SENSE or swallow a command.
"""

import functools
import time

from . import phrases as P
from . import replies as R
from .config import MappingConfig
from .doors import DoorLinks
from .fingerprint import Embedder, orb_features, pack, sharpness, unpack
from .objects import ObjectRooms
from .photos import RoomPhotos
from .recognise import UNKNOWN, Recogniser, RoomModel
from .rooms import Rooms
from .schema import migrate
from .teaching import TeachSession

STOP_WORDS = {"stop", "done", "i'm done", "im done", "that's enough", "thats enough", "enough",
              "finish", "finished", "stop learning", "stop teaching", "ok stop", "okay stop", "i am done"}
CANCEL_WORDS = {"cancel", "cancel that", "never mind", "nevermind", "forget it", "cancel learning"}


def _safe(fn):
    @functools.wraps(fn)
    def inner(self, *a, **kw):
        if not self.enabled:
            return None
        try:
            return fn(self, *a, **kw)
        except Exception as e:          # never let the home map break SENSE
            if fn.__name__ not in self._errors:
                self._errors.add(fn.__name__)
                print(f"🏠 Home map {fn.__name__} error (ignored): {e!r}")
            return None
    return inner


class HomeMap:
    def __init__(self, conn, session_id, cfg=None, clock=time.time, embedder=None):
        self.cfg = cfg or MappingConfig()
        self.session_id = session_id
        self.clock = clock
        self.enabled = True
        self._errors = set()
        migrate(conn)                   # raises SchemaConflict -> create() switches us off
        self.rooms = Rooms(conn, clock)
        self.photos = RoomPhotos(conn, clock)
        self.doors = DoorLinks(conn, clock)
        self.objects = ObjectRooms(conn, clock)
        self.embedder = embedder        # None = corner features + landmarks only
        self.recogniser = Recogniser(self.cfg)
        self.teach = None
        self.pending = []               # (text, time queued, keep even if old)
        self.frames = 0
        self.last_frame = None          # last steady frame, for an on-demand check
        self.last_landmarks = set()
        self.reload_rooms()

    # ------------------------------------------------------------ data
    def reload_rooms(self):
        models = []
        for room in self.rooms.all():
            views = []
            for row in self.photos.for_room(room["id"]):
                if not row["fingerprint"]:
                    continue
                emb, pts, des = unpack(row["fingerprint"])
                if emb is not None and self.embedder is None:
                    emb = None          # can't compare with no model loaded
                views.append({"emb": emb, "pts": pts, "des": des, "landmarks": row["landmarks"]})
            models.append(RoomModel(room["id"], room["name"], views))
        self.recogniser.set_rooms(models)

    def taught_rooms(self):
        return {rid: m.name for rid, m in self.recogniser.rooms.items()}

    def _features(self, frame):
        emb = self.embedder.embed(frame) if self.embedder is not None else None
        pts, des = orb_features(frame)
        return emb, pts, des

    def _landmarks(self, boxes, names):
        """Set of landmark class names in this frame's YOLO boxes (None if YOLO skipped)."""
        if boxes is None:
            return None
        out = set()
        if hasattr(boxes, "cls"):                         # ultralytics Boxes
            pairs = zip(boxes.cls.tolist(), boxes.conf.tolist())
            pairs = ((names[int(c)], float(s)) for c, s in pairs)
        else:                                              # [(name, conf)] in tests
            pairs = boxes
        for name, conf in pairs:
            if conf >= self.cfg.landmark_min_conf and name in self.cfg.landmark_classes:
                out.add(name)
        return out

    def _say(self, text, now, keep=False):
        self.pending.append((text, now, keep))

    # ------------------------------------------------------------ frames
    @_safe
    def observe(self, frame, boxes, names, heading, camera_stable, now=None, can_speak=True):
        now = self.clock() if now is None else now
        self.frames += 1
        landmarks = self._landmarks(boxes, names)
        if landmarks is not None:
            self.last_landmarks = landmarks
        if self.teach is not None:
            sess = self.teach

            def capture():
                emb, pts, des = self._features(frame)
                return {"emb": emb, "pts": pts, "des": des, "landmarks": sorted(self.last_landmarks)}
            ev = sess.feed(heading, now, sharpness(frame), capture)
            if ev is not None:
                self._teach_event(ev, now)
        elif camera_stable and self.recogniser.rooms and self.frames % self.cfg.recognise_every_n == 0:
            emb, pts, des = self._features(frame)
            self.recogniser.update(emb, pts, des, self.last_landmarks, now)
        if camera_stable:
            self.last_frame = frame
        return self._pop(now, can_speak)

    def _pop(self, now, can_speak):
        self.pending = [p for p in self.pending if p[2] or now - p[1] <= 8.0]
        if can_speak and self.pending:
            return self.pending.pop(0)[0]
        return None

    def _teach_event(self, ev, now):
        kind, info = ev
        sess = self.teach
        if kind == "nudge":
            self._say(R.teach_nudge(), now)
        elif kind == "progress":
            self._say(R.teach_halfway(), now)
        elif kind == "done":
            self._say(self._finish_teaching(now), now, keep=True)
        elif kind == "failed":
            self.teach = None
            self._say(R.teach_failed(sess.room_name), now, keep=True)

    def _finish_teaching(self, now):
        sess, self.teach = self.teach, None
        if sess.relearn:
            self.photos.clear(sess.room_id)
        for v in sess.views:
            self.photos.add(sess.room_id, heading_deg=v["heading_deg"],
                            fingerprint=pack(v["emb"], v["pts"], v["des"]),
                            landmarks=v["landmarks"], session_id=self.session_id, now=now)
        self.reload_rooms()
        self.recogniser.assert_room(sess.room_id)
        self.rooms.register(sess.room_name, self.session_id, now)    # you are here now
        print(f"🏠 Learned {sess.room_name}: {len(sess.views)} views, {sess.coverage_deg():.0f} degrees")
        return R.teach_done(sess.room_name, len(sess.views))

    def _start_teaching(self, room, heading, now, relearn=False):
        self.teach = TeachSession(room["id"], room["name"], heading, now, self.cfg, relearn=relearn)
        self.pending = []
        return R.teach_start(room["name"], relearn)

    # ------------------------------------------------------------ voice
    @_safe
    def handle(self, text, now=None, known_names=(), heading=0.0):
        now = self.clock() if now is None else now
        t = P.normalise(text, self.cfg)
        if self.teach is not None and (t in STOP_WORDS or t in CANCEL_WORDS):
            sess = self.teach
            if t in CANCEL_WORDS:
                self.teach = None
                return R.teach_cancelled(sess.room_name)
            kind, _ = sess.stop()
            if kind == "done":
                return self._finish_teaching(now)
            self.teach = None
            return R.teach_failed(sess.room_name)
        cmd = P.parse(text, self.rooms.names(), known_names, self.cfg)
        if cmd is None:
            return None
        print(f"🏠 Room command: {cmd.kind}" + (f" ({cmd.room})" if cmd.room else ""))
        if self.teach is not None and cmd.kind != P.LIST:
            return R.still_teaching(self.teach.room_name)
        if cmd.kind == P.LIST:
            return R.room_list(self.rooms.names())
        if cmd.kind == P.TEACH:
            if not cmd.room:
                return R.teach_missed(cmd.heard)
            room, is_new = self.rooms.register(cmd.room, self.session_id, now)
            if self.photos.count(room["id"]) > 0:
                self.recogniser.assert_room(room["id"])
                return R.already_taught(room["name"])
            return self._start_teaching(room, heading, now)
        if cmd.kind == P.RELEARN:
            if cmd.room:
                room = self.rooms.get(cmd.room)
                if room is None:
                    return R.relearn_unknown(cmd.room)
            else:
                room = self.current_room(now)
                if room is None:
                    return R.relearn_which()
            return self._start_teaching(room, heading, now, relearn=True)
        if cmd.kind == P.WHERE_AM_I:
            return self._where_am_i(now)
        if cmd.kind == P.GO:
            known = self.rooms.get(cmd.room) is not None
            cur = self.current_room(now)
            return R.go(cmd.room, known, cur["name"] if cur else None)
        return None

    def _where_am_i(self, now):
        declared = self.declared_room(now)
        if not self.recogniser.rooms:             # nothing taught by sight yet: Phase 0 answer
            if declared is not None:
                return R.where_am_i(declared["name"], now - declared["last_declared_at"], True)
            return R.where_am_i(None, 0, bool(self.rooms.names()))
        rec = self.recogniser
        if (rec.last_update is None or now - rec.last_update > self.cfg.belief_fresh_s) \
                and self.last_frame is not None:
            emb, pts, des = self._features(self.last_frame)
            rec.update(emb, pts, des, self.last_landmarks, now)
        state, p = rec.best()
        if state is not None and state != UNKNOWN and p >= self.cfg.likely_prob:
            return R.recognised(rec.rooms[state].name, p >= self.cfg.sure_prob and rec.verified(state))
        if state == UNKNOWN and p >= self.cfg.likely_prob and declared is None:
            return R.unknown_place()
        if declared is not None:
            return R.not_sure(declared["name"], now - declared["last_declared_at"])
        if self.last_frame is None:
            return R.no_camera_yet()
        return R.not_sure()

    # ------------------------------------------------------------ state
    def declared_room(self, now):
        """The room the user named in THIS run, if recent enough; else None."""
        room = self.rooms.last_declared(self.session_id)
        if room is None or now - room["last_declared_at"] > self.cfg.declared_room_valid_s:
            return None
        return room

    def current_room(self, now):
        """Recognised room if SENSE is sure, else the room the user last named."""
        state, p = self.recogniser.best()
        if state is not None and state != UNKNOWN and p >= self.cfg.sure_prob and self.recogniser.verified(state):
            return self.rooms.get_by_id(state)
        return self.declared_room(now)


class _Disabled:
    """Stands in when the home map can't start: every call does nothing."""
    enabled = False

    def handle(self, *a, **kw):
        return None

    observe = handle


def create(session_id, conn=None, cfg=None, download=True):
    """Make the home map on memory_db's connection; never raises."""
    try:
        if conn is None:
            import memory_db
            conn = memory_db._get_conn()
        embedder = Embedder.create(download=download)
        home = HomeMap(conn, session_id, cfg, embedder=embedder)
        n, taught = len(home.rooms.names()), len(home.recogniser.rooms)
        print(f"🏠 Home map on ({n} room{'s' if n != 1 else ''} registered, {taught} learned by sight).")
        return home
    except Exception as e:
        print(f"🏠 Home map off ({e!r})")
        return _Disabled()
