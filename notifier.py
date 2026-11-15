"""
notifier.py - Deciding WHAT to say on its own, and WHEN (Roadmap E3)

A blind user cannot "glance" at a screen, so every spoken word costs attention.
This module turns raw events into a small number of useful announcements.

Rules (each is a classic technique, deliberately simple to explain):
  * PRIORITIES  P1 safety > P2 personal (your items, people you know) > P3 generic.
  * LEVELS      quiet = P1 only, normal = P1+P2 (default), verbose = everything.
  * COOLDOWN    the same item is not announced again within PER_ITEM_COOLDOWN_S
                (stops "cup moved, cup moved, cup moved").
  * MIN GAP     at least MIN_GAP_S between announcements (a simple rate limit).
  * NO TALKING OVER  a P2/P3 waits while SENSE is already speaking (e.g. an answer);
                only P1 may interrupt.
  * COALESCING  up to two waiting messages are merged into one sentence.
  * STALENESS   a message that waited longer than MAX_WAIT_S is dropped:
                old news is noise.
  * CAMERA MOVING  only P1 is spoken while the user is turning/walking.

It runs inside the main loop (tick() once per frame). The speech itself is the
macOS `say` process, so nothing here blocks and no thread is needed.
"""

import heapq
import itertools
import time

P1, P2, P3 = 1, 2, 3
LEVELS = {"quiet": P1, "normal": P2, "verbose": P3}


class Notifier:
    PER_ITEM_COOLDOWN_S = 60.0
    MIN_GAP_S = 4.0
    MAX_WAIT_S = 10.0
    P1_COOLDOWN_S = 6.0

    def __init__(self, speak_fn, is_busy_fn, level="normal", clock=time.time):
        self.speak_fn = speak_fn
        self.is_busy_fn = is_busy_fn
        self.level = level if level in LEVELS else "normal"
        self.clock = clock
        self.pending = []           # heap of (priority, order, created, key, text)
        self._order = itertools.count()
        self.last_spoken_key_at = {}
        self.last_spoken_at = -1e9

    def set_level(self, level):
        if level in LEVELS:
            self.level = level

    def push(self, priority, key, text):
        if priority > LEVELS[self.level]:
            return False
        now = self.clock()
        cooldown = self.P1_COOLDOWN_S if priority == P1 else self.PER_ITEM_COOLDOWN_S
        if now - self.last_spoken_key_at.get(key, -1e9) < cooldown:
            return False
        # replace an older pending message about the same key (keep newest)
        self.pending = [p for p in self.pending if p[3] != key]
        heapq.heapify(self.pending)
        heapq.heappush(self.pending, (priority, next(self._order), now, key, text))
        return True

    def tick(self, camera_moving=False, hold=False):
        """Call once per frame. Returns the sentence spoken, or None.

        hold=True while the microphone is open: only P1 (safety) may speak,
        everything else waits, so SENSE never talks into its own recording."""
        now = self.clock()
        if len(self.last_spoken_key_at) > 200:     # forget keys whose cooldown is long over
            horizon = max(self.PER_ITEM_COOLDOWN_S, self.P1_COOLDOWN_S)
            self.last_spoken_key_at = {k: t for k, t in self.last_spoken_key_at.items() if now - t < horizon}
        self.pending = [p for p in self.pending if now - p[2] <= self.MAX_WAIT_S]
        heapq.heapify(self.pending)
        if not self.pending:
            return None
        top_priority = self.pending[0][0]
        if top_priority != P1:
            if hold or camera_moving or self.is_busy_fn() or now - self.last_spoken_at < self.MIN_GAP_S:
                return None
        batch = [heapq.heappop(self.pending)]
        if self.pending and self.pending[0][0] == top_priority and top_priority != P1:
            batch.append(heapq.heappop(self.pending))
        sentence = " ".join(text if text.endswith(".") else text + "." for _, _, _, _, text in batch)
        self.speak_fn(sentence)
        self.last_spoken_at = now
        for _, _, _, key, _ in batch:
            self.last_spoken_key_at[key] = now
        return sentence
