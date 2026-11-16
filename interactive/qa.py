"""
interactive/qa.py - the InteractiveQA facade: the only object sense_master uses.

Order of work for one question (all on the main thread, between two frames):

  1. If an existing SENSE command owns the words -> return None (old code runs).
  2. Rule-based answer types (parser.py + responders.py)     -> speak, done.
  3. A follow-up with "it" for the existing where/when code   -> rewritten text.
  4. Anything else falls through to the existing code. Only where that code
     would say "I did not catch which item", sense_master asks free_form():
     the local AI model gets a fact sheet and its answer is fact-checked.

Speaking: answers go through SENSE's own speak(). If a SAFETY alert (P1) is
being spoken at that moment, the answer waits (at most P1_HOLD_MAX_S), so a
question can never cut off "Careful, chair ahead".
"""
import time
from dataclasses import dataclass
from types import SimpleNamespace

from . import config as C
from . import facts as F
from . import guard as G
from . import parser as P
from . import responders as R
from . import text as T
from .context import DialogueContext
from .snapshot import SnapshotBuilder

RESPONDERS = {"TIME": None, "DATE": None, "READ": R.read_answer, "COLOUR": R.colour_answer,
              "COUNT": R.count_answer, "SIDE": R.side_answer, "DESCRIBE": R.describe_answer,
              "PAST": R.past_answer}


@dataclass
class Reply:
    kind: str        # "spoken" (answered) | "rewrite" (give `text` to the existing code)
    text: str
    qtype: str = ""


class InteractiveQA:
    def __init__(self, hooks, llm=None, clock=time.time, log=print):
        self.h = hooks if not isinstance(hooks, dict) else SimpleNamespace(**hooks)
        self.llm = llm
        self.clock = clock
        self.log = log
        self.snap = SnapshotBuilder()
        self.ctx = DialogueContext()
        self.held = None          # (text, since) waiting for a safety alert to finish
        self.job = None           # {"job", "facts", "question", "start", "filler"}
        self.stats = {"rules": 0, "rewrite": 0, "declined": 0, "llm": 0, "llm_rejected": 0,
                      "llm_timeout": 0, "errors": 0}
        if llm is not None:
            llm.start()

    # ------------------------------------------------------------ per frame
    def see(self, *args):
        try:
            self.snap.see(*args)
        except Exception as e:      # never let Path B disturb the main loop
            self._error("see", e)

    def frame_done(self, now, frame_w, frame_h, gray=None):
        try:
            self.snap.frame_done(now, frame_w, frame_h, gray)
        except Exception as e:
            self._error("frame_done", e)

    def poll(self, now=None):
        """Once per frame: held answers, the AI model's reply, start-up checks."""
        now = self.clock() if now is None else now
        try:
            if self.llm is not None:
                self.llm.tick()
            if self.held is not None:
                text, since = self.held
                if not self._must_wait() or now - since > C.P1_HOLD_MAX_S:
                    self.held = None
                    if not self.h.is_listening():
                        self.h.speak(text)
            if self.job is not None:
                self._poll_job(now)
        except Exception as e:
            self._error("poll", e)

    # ------------------------------------------------------------ questions
    def handle(self, user_text, now=None):
        now = self.clock() if now is None else now
        try:
            self._cancel_job()
            self.held = None
            existing = self.h.classify(user_text)
            ocr_owns = bool(self.h.ocr_owns(user_text))
            q = P.parse(user_text, existing, ocr_owns, self.h.find_target)
            if q is None:
                if not ocr_owns and existing not in P.GUARD_INTENTS:
                    rw = P.rewrite_follow_up(user_text, self.ctx.phrase(now), existing)
                    if rw:
                        self.stats["rewrite"] += 1
                        self.log(f"💬 Follow-up: '{user_text}' -> '{rw}'")
                        return Reply("rewrite", rw, "FOLLOW_UP")
                self._note_subject(user_text, now)
                self.stats["declined"] += 1
                return None
            self.log(f"💬 Question type: {q.kind}" + (f" ({q.cls or q.name or ''}"
                                                       f"{', ' + q.side if q.side else ''})" if (q.cls or q.name or q.side) else ""))
            if q.kind == "TIME":
                answer = R.time_answer(now)
            elif q.kind == "DATE":
                answer = R.date_answer(now)
            else:
                answer = RESPONDERS[q.kind](q, self.snap.current, now, self.h, self.ctx)
            if isinstance(answer, R.Rewrite):
                self.stats["rewrite"] += 1
                return Reply("rewrite", str(answer), q.kind)
            if q.kind != "READ" and (q.cls or q.name) and self.ctx.get(now) is None:
                self.ctx.set(now, cls=q.cls, name=q.name)
            self.stats["rules"] += 1
            self.say(answer, now)
            return Reply("spoken", answer, q.kind)
        except Exception as e:
            self._error("handle", e)
            return None

    def free_form(self, user_text, now=None):
        """The existing code found no item: let the local model try. True = handled."""
        now = self.clock() if now is None else now
        try:
            if self.llm is None or not self.llm.ready:
                return False
            G.KNOWN_NAME_HINTS.update(p.lower() for p in self.h.known_people())
            G.KNOWN_NAME_HINTS.update(n.lower() for n in self.h.unverified())
            facts = F.build(self.snap.current, now, self.h)
            job = self.llm.ask(F.SYSTEM_PROMPT, F.prompt(facts, user_text))
            self.job = {"job": job, "facts": facts, "question": user_text, "start": now, "filler": False}
            self.stats["llm"] += 1
            self.log(f"💬 Asking the local AI model ({self.llm.model}): '{user_text}'")
            return True
        except Exception as e:
            self._error("free_form", e)
            return False

    def say(self, text, now=None):
        now = self.clock() if now is None else now
        text = T.sentence(text)
        if self._must_wait():
            self.held = (text, now)       # spoken by poll() when the safety alert ends
            self.log(f"💬 Answer waits for a safety alert: '{text}'")
        else:
            self.h.speak(text)

    # ------------------------------------------------------------ internals
    def _must_wait(self):
        return bool(self.h.p1_talking()) or bool(self.h.is_listening())

    def _poll_job(self, now):
        j = self.job
        if j["job"].done():
            self.job = None
            raw, err = j["job"].result()
            took = now - j["start"]
            if err is not None:
                self.log(f"💬 Local AI model failed after {took:.1f}s: {err}")
                self.say(G.FALLBACK, now)
                return
            answer = G.clean(raw)
            ok, why = G.check(answer, j["facts"])
            self.log(f"💬 Model said ({took:.1f}s): '{raw}' -> {'OK' if ok else 'REJECTED: ' + why}")
            if not ok:
                self.stats["llm_rejected"] += 1
                answer = G.FALLBACK
            self.say(answer, now)
            return
        if now - j["start"] > C.LLM_TIMEOUT_S + 0.5:
            self.stats["llm_timeout"] += 1
            self._cancel_job()
            self.say("Sorry, that took too long. Please ask in a shorter way.", now)
            return
        if not j["filler"] and now - j["start"] > C.LLM_FILLER_AFTER_S and not self.h.is_speaking() \
                and not self._must_wait():
            j["filler"] = True
            self.h.speak("One moment.")

    def _cancel_job(self):
        if self.job is not None:
            self.job["job"].cancel()
            self.job = None

    def _note_subject(self, user_text, now):
        """Remember what a question handled by the EXISTING code was about, so that
        a later "what colour is it?" knows what "it" is."""
        t = T.normalise(user_text)
        name = self.h.find_target(user_text)
        cls = T.class_in(t)
        if name:
            n = name.lower()
            self.ctx.set(now, cls=T.CLASS_WORDS.get(n.split()[-1]) or cls,
                         name=None if T.CLASS_WORDS.get(n) else n)
        elif cls:
            self.ctx.set(now, cls=cls)

    def _error(self, where, e):
        self.stats["errors"] += 1
        self.log(f"💬 Interactive layer error in {where}: {type(e).__name__}: {e} (SENSE goes on as before)")

    def close(self):
        self._cancel_job()
        if self.llm is not None:
            self.llm.close()
