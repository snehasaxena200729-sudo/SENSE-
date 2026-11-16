"""Unit tests for interactive/ (Path B: questions about the surroundings).

    python -m unittest tests.test_interactive -v

No camera, mic, YOLO or Ollama needed: scenes are built by hand, the colour
classifier is the real one from sense_master.py, and the local AI model is a
tiny fake HTTP server that speaks Ollama's API (so the real curl plumbing runs).
"""
import http.server
import json
import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interactive import InteractiveQA, OllamaLLM, guard, parser  # noqa: E402
from interactive.colour import read_colour  # noqa: E402
import answers  # noqa: E402
from interactive_fakes import BGR, FakeMemory, crop_of, get_dominant_color_name, make_hooks  # noqa: E402

FW, FH, NOW = 1280, 720, 1_763_300_000.0
BRIGHT = np.full((90, 160), 140, np.uint8)
DARK = np.full((90, 160), 20, np.uint8)


def scene(qa, items, now=NOW, gray=BRIGHT):
    for tid, (cls, label, person, box, crop, dist) in enumerate(items, 1):
        qa.see(tid, cls, label, person, box, crop, dist, None, FW, FH)
    qa.frame_done(now, FW, FH, gray)


def ask(qa, text, now=NOW):
    r = qa.handle(text, now)
    return None if r is None else (r.kind, r.text)


LEFT, MID, RIGHT = (100, 300, 200, 450), (600, 300, 700, 450), (1000, 300, 1100, 450)


class TestGuard(unittest.TestCase):
    """Existing commands must reach the existing code: Path B returns None."""
    EXISTING = ["find my phone", "forget my bottle", "rename my bottle to medicine", "this is Sneha",
                "this is my blue bottle", "be quiet", "speak slower", "what changed", "who was here",
                "what did you see recently", "where is my phone", "where is the door", "how far is the door",
                "where is Sneha", "what do you see", "describe the room", "read this", "stop reading",
                "keep reading", "when did you see my phone", "where was my bottle before", "is sneha",
                "stop looking", "what's around me", "what time did you see the cup"]

    def test_existing_commands_untouched(self):
        h = make_hooks(known=["Sneha"])
        qa = InteractiveQA(h, log=lambda *_: None)
        for text in self.EXISTING:
            with self.subTest(text=text):
                self.assertIsNone(qa.handle(text, NOW))
        self.assertEqual(h.spoken, [])


class TestParser(unittest.TestCase):
    def p(self, text):
        return parser.parse(text, answers.classify_intent(text), False, None)

    def test_types(self):
        cases = {"What colour is the bottle?": "COLOUR", "what color is it": "COLOUR",
                 "How many chairs are in front of me?": "COUNT", "how many people are around me": "COUNT",
                 "Is there anything on my right?": "SIDE", "what is on my left": "SIDE",
                 "What was previously in front of me?": "PAST", "what is behind me": "PAST",
                 "Describe the object in front of me.": "DESCRIBE", "what am I holding": "DESCRIBE",
                 "What is today's date?": "DATE", "what day is it": "DATE", "What time is it?": "TIME",
                 "What does this medicine label say?": "READ", "is it safe to walk forward": "SIDE",
                 "is the bottle blue": "COLOUR"}
        for text, kind in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.p(text).kind, kind)

    def test_slots(self):
        q = self.p("how many chairs are on my left")
        self.assertEqual((q.cls, q.side), ("chair", "left"))
        self.assertEqual(self.p("what colour are my keys").undetectable, "keys")
        self.assertTrue(self.p("is it safe to walk forward").safety)
        self.assertEqual(self.p("is the cup red").colour, "red")

    def test_follow_up_rewrite(self):
        self.assertEqual(parser.rewrite_follow_up("Where is it?", "the bottle", "WHERE"), "where is the bottle")
        self.assertIsNone(parser.rewrite_follow_up("where is it", None, "WHERE"))
        self.assertIsNone(parser.rewrite_follow_up("find it", "the bottle", "FIND"))


class TestColour(unittest.TestCase):
    def test_clear_colours(self):
        for name in BGR:
            r = read_colour(crop_of(name), get_dominant_color_name, MID, FW, FH, 140)
            self.assertEqual((r.status, r.name), ("clear", name), name)

    def test_honesty(self):
        self.assertEqual(read_colour(crop_of("blue", 15, 15), get_dominant_color_name).status, "too_small")
        self.assertEqual(read_colour(crop_of("blue"), get_dominant_color_name, (0, 300, 60, 380), FW, FH).status,
                         "partial")
        self.assertEqual(read_colour(crop_of("blue", dark=True), get_dominant_color_name, MID, FW, FH, 20).status,
                         "too_dark")
        # a black object in a bright room is black, not "too dark"
        self.assertEqual(read_colour(crop_of("black"), get_dominant_color_name, MID, FW, FH, 140).name, "black")
        r = read_colour(crop_of("blue", second="green", second_share=0.5, w=100), get_dominant_color_name, MID, FW, FH, 140)
        self.assertEqual(r.status, "ambiguous")
        self.assertEqual({r.name, r.alternative}, {"blue", "green"})


class TestAnswers(unittest.TestCase):
    def setUp(self):
        self.mem = FakeMemory()
        self.h = make_hooks(memory=self.mem, known=["Sneha"], legacy=["Rahul"])
        self.qa = InteractiveQA(self.h, log=lambda *_: None)

    def test_colour_and_follow_up(self):
        scene(self.qa, [("bottle", "Blue Bottle", None, MID, crop_of("blue"), "about 2 to 3 feet away")])
        self.assertEqual(ask(self.qa, "What colour is the bottle?"), ("spoken", "The bottle looks blue."))
        self.assertEqual(ask(self.qa, "Where is it?"), ("rewrite", "where is the blue bottle"))
        self.assertEqual(ask(self.qa, "what colour is it")[1], "The bottle looks blue.")

    def test_two_bottles_asks_which(self):
        scene(self.qa, [("bottle", "Blue Bottle", None, LEFT, crop_of("blue"), "x"),
                        ("bottle", "White Bottle", None, RIGHT, crop_of("white"), "x")])
        text = ask(self.qa, "what colour is the bottle")[1]
        self.assertIn("two bottles", text)
        self.assertIn("Which one", text)

    def test_colour_not_visible_uses_memory(self):
        scene(self.qa, [])
        self.mem.add("white cup", NOW - 240)
        self.assertEqual(ask(self.qa, "what colour is the cup")[1],
                         "I can't see the cup right now. When I saw it 4 minutes ago, it looked white.")

    def test_count_no_double_counting_and_names(self):
        scene(self.qa, [("chair", "Chair", None, LEFT, crop_of("black"), "x"),
                        ("chair", "Chair", None, MID, crop_of("black"), "x"),
                        ("person", "Sneha", "Sneha", RIGHT, crop_of("gray"), "x"),
                        ("person", "Person", None, MID, crop_of("gray"), "x")])
        self.assertIn("two chairs", ask(self.qa, "how many chairs are in front of me")[1])
        self.assertEqual(ask(self.qa, "how many people are around me")[1],
                         "I can see two people: Sneha and one person I don't recognise.")

    def test_never_names_unknown_person(self):
        scene(self.qa, [("person", "Person", None, MID, crop_of("gray"), "about 6 feet away")])
        text = ask(self.qa, "describe the person in front of me")[1]
        self.assertIn("a person", text.lower())
        self.assertIn("don't recognise", text)
        self.assertNotIn("Sneha", text)

    def test_side_and_safety(self):
        scene(self.qa, [("cup", "White Cup", None, RIGHT, crop_of("white"), "within arm's reach"),
                        ("chair", "Chair", None, MID, crop_of("black"), "about 3 to 4 feet away")])
        self.assertEqual(ask(self.qa, "Is there anything on my right?")[1],
                         "On your right, I can see a cup, within arm's reach.")
        safety = ask(self.qa, "is it safe to walk forward")[1]
        self.assertIn("chair", safety)
        self.assertIn("can't promise", safety)
        self.assertIn("No, I don't see a chair on your left", ask(self.qa, "is there a chair on my left")[1])

    def test_stale_frame_is_not_now(self):
        scene(self.qa, [("chair", "Chair", None, MID, crop_of("black"), "x")], now=NOW - 5)
        self.assertIn("haven't had a clear look", ask(self.qa, "how many chairs")[1])

    def test_past_direction_this_session_only(self):
        scene(self.qa, [])
        self.mem.add("my phone", NOW - 180, world_angle=3)
        self.mem.add("laptop", NOW - 60, world_angle=10, session_id="OLD")    # earlier run: no angle meaning
        self.mem.add("rahul", NOW - 30, world_angle=0)                       # clothing-colour sighting: hidden
        text = ask(self.qa, "what was previously in front of me")[1]
        self.assertIn("your phone 3 minutes ago", text)
        self.assertNotIn("laptop", text)
        self.assertNotIn("rahul", text.lower())

    def test_behind_and_turning(self):
        scene(self.qa, [])
        self.mem.add("bed", NOW - 100, world_angle=0)
        self.h.state["heading"] = 180.0          # user turned around
        self.assertIn("the bed", ask(self.qa, "what is behind me")[1])

    def test_clock_and_read(self):
        self.assertTrue(ask(self.qa, "what time is it")[1].startswith("It is "))
        self.assertTrue(ask(self.qa, "what's the date today")[1].startswith("Today is "))
        self.assertEqual(ask(self.qa, "what does this medicine label say")[1], "Reading. Hold it steady.")

    def test_waits_for_safety_alert(self):
        scene(self.qa, [("cup", "Cup", None, MID, crop_of("red"), "x")])
        self.h.state["p1"] = True
        ask(self.qa, "what colour is the cup")
        self.assertEqual(self.h.spoken, [])           # held, not talking over "Careful ..."
        self.h.state["p1"] = False
        self.qa.poll(NOW + 0.5)
        self.assertEqual(self.h.spoken, ["The cup looks red."])

    def test_errors_never_escape(self):
        self.h.find_target = lambda t: 1 / 0
        self.assertIsNone(self.qa.handle("what colour is the cup", NOW))
        self.qa.see("bad")                             # wrong arguments
        self.qa.poll(NOW)
        self.assertGreaterEqual(self.qa.stats["errors"], 2)


class TestFactGuard(unittest.TestCase):
    FACTS = ("Visible now (camera, this second):\n- the chair: in front of you, about 3 to 4 feet away, looks black\n"
             "- Sneha (recognised by face): on your right\nCounts in view: 1 chair, 1 person\n"
             "Not visible now, remembered from earlier:\n- your phone: last seen 4 minutes ago, on your left")

    def test_accepts_grounded(self):
        for a in ["I can see a black chair in front of you.", "Sneha is on your right.",
                  "I last saw your phone 4 minutes ago, on your left.", "I can't detect keys.",
                  "I don't know."]:
            self.assertTrue(guard.check(a, self.FACTS)[0], a)

    def test_rejects_invented(self):
        for a in ["There is a red chair.", "I can see two chairs.", "Rahul is here.", "Your keys are on the table.",
                  "The path ahead is clear.", "It is 7 feet away.", "I see a laptop.", "There is a cup next to you."]:
            self.assertFalse(guard.check(a, self.FACTS)[0], a)


class FakeOllama(http.server.BaseHTTPRequestHandler):
    reply = "I can see a black chair in front of you."
    delay = 0.0
    models = ["llama3.2:3b"]

    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"models": [{"name": n} for n in FakeOllama.models]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.last = req
        time.sleep(FakeOllama.delay if req.get("prompt") else 0)
        self._send({"response": FakeOllama.reply if req.get("prompt") else "", "done": True})


@unittest.skipUnless(os.path.exists("/usr/bin/curl") or os.path.exists("/bin/curl"), "needs curl")
class TestLocalModel(unittest.TestCase):
    """The real curl subprocess against a fake Ollama (the tests' server thread is test-only)."""

    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.host = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def run_until(self, qa, h, cond, limit=6.0):
        t0 = time.time()
        while time.time() - t0 < limit and not cond():
            qa.poll(time.time())
            time.sleep(0.02)

    def make(self, models=("llama3.2:3b",)):
        FakeOllama.models = list(models)
        h = make_hooks(known=["Sneha"])
        log = []
        llm = OllamaLLM(host=self.host, timeout_s=3, log=log.append)
        qa = InteractiveQA(h, llm=llm, log=log.append)
        self.run_until(qa, h, lambda: llm.state != "checking")
        return qa, h, llm, log

    def test_grounded_answer_spoken(self):
        qa, h, llm, _ = self.make()
        self.assertEqual(llm.state, "ready")
        scene(qa, [("chair", "Chair", None, MID, crop_of("black"), "about 3 to 4 feet away")], now=time.time())
        FakeOllama.reply, FakeOllama.delay = "I can see a black chair in front of you.", 0
        self.assertTrue(qa.free_form("can I sit somewhere", time.time()))
        self.run_until(qa, h, lambda: h.spoken)
        self.assertEqual(h.spoken, ["I can see a black chair in front of you."])
        self.assertIn("chair", FakeOllama.last["prompt"])
        self.assertEqual(FakeOllama.last["options"]["temperature"], 0)

    def test_hallucination_replaced(self):
        qa, h, _, _ = self.make()
        scene(qa, [], now=time.time())
        FakeOllama.reply = "Your keys are on the red table next to Rahul."
        qa.free_form("where can I sit", time.time())
        self.run_until(qa, h, lambda: h.spoken)
        self.assertEqual(h.spoken, [guard.FALLBACK])

    def test_slow_model_filler_then_answer(self):
        qa, h, _, _ = self.make()
        scene(qa, [], now=time.time())
        FakeOllama.reply, FakeOllama.delay = "I don't know.", 1.8
        qa.free_form("tell me something", time.time())
        self.run_until(qa, h, lambda: len(h.spoken) >= 2)
        self.assertEqual(h.spoken, ["One moment.", "I don't know."])
        FakeOllama.delay = 0

    def test_model_missing_means_rules_only(self):
        qa, h, llm, log = self.make(models=["other:1b"])
        self.assertEqual(llm.state, "no_model")
        self.assertFalse(qa.free_form("anything", time.time()))
        self.assertTrue(any("ollama pull" in l for l in log))

    def test_no_server(self):
        h = make_hooks()
        llm = OllamaLLM(host="http://127.0.0.1:9", timeout_s=2, log=lambda *_: None)
        qa = InteractiveQA(h, llm=llm, log=lambda *_: None)
        self.run_until(qa, h, lambda: llm.state != "checking")
        self.assertEqual(llm.state, "no_server")


if __name__ == "__main__":
    unittest.main()
