"""
recognise.py - which taught room does the camera see?

Three clues per check, combined into one score per room:

1. Image embedding, k-nearest-neighbour style: the room whose saved views are
   most similar (cosine) to the live view. With 2+ rooms we first subtract the
   average view of the home ("centering"): every indoor picture shares a lot
   (walls, ceilings, light), and what is left is what makes one room
   different from another. Measured on real indoor photos this separates
   same-room from other-room pairs far better (see docs/HOME_MAP.md).
2. ORB check on the two best rooms: corners matched between the live view and
   that room's closest saved views, then RANSAC keeps only matches that agree
   on one camera geometry (fundamental matrix). Many agreeing matches = the
   same spot, so that room's score is boosted.
3. Landmarks: things YOLO saw while you taught the room (refrigerator, bed,
   toilet...). Seeing one now makes the rooms where it was seen more likely
   (a small Naive Bayes factor, clipped so it can never decide on its own).

The scores are not trusted frame by frame. A Bayes filter (the same idea as
a Hidden Markov Model) keeps a probability for every room plus "a place I
don't know": each check multiplies in the new evidence, and between checks
you most likely stay where you are. So one odd frame cannot flip the answer.
"""

import math

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

UNKNOWN = "__unknown__"


class RoomModel:
    """All saved views of one room, ready for matching (kept in memory)."""

    def __init__(self, room_id, name, views):
        self.room_id, self.name = room_id, name
        embs = [v["emb"] for v in views if v.get("emb") is not None]
        self.embs = np.array(embs, np.float32) if embs else np.zeros((0, 0), np.float32)
        self.orb = [(v["pts"], v["des"]) for v in views]
        self.n_views = len(views)
        counts = {}
        for v in views:
            for lm in set(v.get("landmarks") or ()):
                counts[lm] = counts.get(lm, 0) + 1
        self.landmark_counts = counts


class Recogniser:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rooms = {}            # room_id -> RoomModel
        self.mean = None           # home's average embedding (for centering)
        self.belief = {}           # state -> probability; states = room ids + UNKNOWN
        self.last_update = None
        self.last_scores = {}
        self._bf = None
        self.verified_at = {}      # room_id -> numbers of the last checks that clearly showed it
        self.last_check_no = 0

    # ------------------------------------------------------------ rooms
    def set_rooms(self, models):
        self.rooms = {m.room_id: m for m in models if m.n_views}
        per_room = [m.embs.mean(0) for m in self.rooms.values() if m.embs.size]
        self.mean = np.mean(per_room, 0) if len(per_room) >= 2 else None
        old = self.belief
        states = list(self.rooms) + [UNKNOWN]
        self.belief = {s: old.get(s, 0.0) for s in states}
        total = sum(self.belief.values())
        if total <= 1e-9:
            self.belief = {s: 1.0 / len(states) for s in states}
        else:
            self.belief = {s: p / total for s, p in self.belief.items()}

    def reset(self):
        states = list(self.rooms) + [UNKNOWN]
        self.belief = {s: 1.0 / len(states) for s in states}
        self.last_update = None

    def assert_room(self, room_id, strength=0.9):
        """The user just told us (or we just taught) the room: trust it."""
        if room_id not in self.belief:
            return
        self.verified_at[room_id] = [self.last_check_no - k for k in (3, 2, 1, 0)]
        others = (1.0 - strength) / max(1, len(self.belief) - 1)
        self.belief = {s: (strength if s == room_id else others) for s in self.belief}

    # ------------------------------------------------------------ evidence
    def _centered(self, v):
        if self.mean is None:
            return v
        c = v - self.mean
        return c / (np.linalg.norm(c) + 1e-9)

    def embedding_scores(self, emb):
        """room_id -> best similarity of the live view to that room's views."""
        out = {}
        if emb is None:
            return out
        q = self._centered(emb)
        for rid, m in self.rooms.items():
            if not m.embs.size:
                continue
            g = m.embs
            if self.mean is not None:
                g = g - self.mean
                g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
            sims = g @ q
            top = np.sort(sims)[-2:]
            out[rid] = float(top.mean()) if len(top) == 2 and self.cfg.knn_two else float(top.max())
        return out

    def orb_inliers(self, pts, des, room_id, max_views=3, emb=None):
        """Best geometric-match count between the live view and this room's views."""
        m = self.rooms[room_id]
        if des is None or len(des) < 8:
            return 0
        if self._bf is None:
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        order = list(range(len(m.orb)))
        if emb is not None and m.embs.size and len(m.embs) == len(m.orb):
            order = list(np.argsort(-(m.embs @ emb)))
        best = 0
        for i in order[:max_views]:
            p2, d2 = m.orb[i]
            if d2 is None or len(d2) < 8:
                continue
            pairs = self._bf.knnMatch(des, d2, k=2)
            good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
            # one match per saved corner: many live corners matching the SAME saved
            # corner (repeated texture, flat walls) is not evidence of the same spot
            used, unique = set(), []
            for g in sorted(good, key=lambda g: g.distance):
                if g.trainIdx not in used:
                    used.add(g.trainIdx)
                    unique.append(g)
            if len(unique) < 8:
                continue
            src = np.float32([pts[g.queryIdx] for g in unique])
            dst = np.float32([p2[g.trainIdx] for g in unique])
            if np.ptp(src[:, 0]) < 40 or np.ptp(src[:, 1]) < 30:
                continue                      # all matches in one small patch: degenerate
            _, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 2.0, 0.99)
            if mask is not None:
                n = int(mask.sum())
                if n >= 0.4 * len(unique):    # most matches must agree on one geometry
                    best = max(best, n)
        return best

    def landmark_factor(self, room_id, seen):
        """Naive Bayes ratio P(seen | room) / P(seen | any room), clipped."""
        if not seen or len(self.rooms) < 2:
            return 1.0
        m = self.rooms[room_id]
        total_views = sum(r.n_views for r in self.rooms.values())
        f = 1.0
        for lm in seen:
            p_room = (m.landmark_counts.get(lm, 0) + 0.5) / (m.n_views + 1.0)
            p_any = (sum(r.landmark_counts.get(lm, 0) for r in self.rooms.values()) + 0.5) / (total_views + 1.0)
            f *= p_room / p_any
        lo, hi = self.cfg.landmark_clip
        return float(min(hi, max(lo, f)))

    # ------------------------------------------------------------ update
    def likelihoods(self, emb, pts, des, landmarks):
        cfg = self.cfg
        centred = self.mean is not None
        beta = cfg.beta_centred if centred else cfg.beta_raw
        tau = cfg.unknown_sim_centred if centred else cfg.unknown_sim_raw
        sims = self.embedding_scores(emb)
        L = {}
        for rid in self.rooms:
            s = sims.get(rid)
            L[rid] = math.exp(beta * (s - tau)) if s is not None else 1.0
        # ORB check on the two most likely rooms
        inliers = {}
        ranked = sorted(self.rooms, key=lambda r: -(L[r] * self.belief.get(r, 0.0)))[:2]
        for rid in ranked:
            n = self.orb_inliers(pts, des, rid, emb=emb)
            inliers[rid] = n
            if n >= cfg.orb_min_inliers:
                L[rid] *= min(cfg.orb_max_boost, 1.0 + n / float(cfg.orb_min_inliers))
        for rid in self.rooms:
            L[rid] *= self.landmark_factor(rid, landmarks)
        lo, hi = cfg.likelihood_clip   # one check can shift the odds only so far: frames
        L = {r: min(hi, max(lo, v)) for r, v in L.items()}   # in a row are not independent
        L[UNKNOWN] = 1.0          # the reference: a room must beat "a place I don't know"
        if emb is None and not any(n >= cfg.orb_min_inliers for n in inliers.values()):
            L = {s: 1.0 for s in L}  # no image model and no corner match: no evidence
        self.last_scores = {"sims": sims, "inliers": inliers}
        sure_sim = cfg.sure_sim_centred if centred else cfg.sure_sim_raw
        for rid in self.rooms:
            ok = inliers.get(rid, 0) >= cfg.orb_min_inliers or (sims.get(rid) or -1) >= sure_sim
            if ok:
                hist = self.verified_at.setdefault(rid, [])
                hist.append(self.last_check_no)
                del hist[:-4]
        return L

    def update(self, emb, pts, des, landmarks, now):
        """One check. Returns (best state, its probability)."""
        if not self.rooms:
            return None, 0.0
        self.last_check_no += 1
        L = self.likelihoods(emb, pts, des, landmarks)
        n = len(self.belief)
        stay = self.cfg.stay_prob
        move = (1.0 - stay) / max(1, n - 1)
        prior = {s: stay * p + move * (1.0 - p) for s, p in self.belief.items()}
        post = {s: prior[s] * L.get(s, 1.0) for s in prior}
        z = sum(post.values()) or 1.0
        self.belief = {s: max(1e-6, p / z) for s, p in post.items()}
        z = sum(self.belief.values())
        self.belief = {s: p / z for s, p in self.belief.items()}
        self.last_update = now
        return self.best()

    def verified(self, room_id, within_checks=4, needed=3):
        """Did at least 3 of the last 4 checks (about 3 seconds) show this room
        CLEARLY (strong image match or corner geometry)? Only then may SENSE
        say "You are in ..." - a glimpse through a doorway is never enough."""
        hist = self.verified_at.get(room_id, [])
        return sum(1 for n in hist if self.last_check_no - n < within_checks) >= needed

    def best(self):
        if not self.belief:
            return None, 0.0
        s = max(self.belief, key=self.belief.get)
        return s, self.belief[s]
