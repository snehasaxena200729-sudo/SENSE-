"""voice_input.py: the SPACE -> record -> Whisper -> queue path, with a fake
microphone and a fake Whisper (no audio hardware in the test machine)."""
import os
import queue
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import answers  # noqa: E402
import voice_input as vi  # noqa: E402

PROMPT = "Sneha, Mummy, water bottle, medicine bottle, phone, laptop, rename."
OLD_PROMPT = "Sneha, Mummy, water bottle, medicine bottle, phone, laptop, rename, forget."


class FakeSR:
    class WaitTimeoutError(Exception):
        pass

    class Microphone:
        opened = 0
        open_now = 0

        def __init__(self, device_index=None):
            self.device_index = device_index

        def __enter__(self):
            FakeSR.Microphone.opened += 1
            FakeSR.Microphone.open_now += 1
            return self

        def __exit__(self, *a):
            FakeSR.Microphone.open_now -= 1

        @staticmethod
        def list_microphone_names():
            return ["MacBook Air Microphone", "Camo Microphone", None]


class FakeRecognizer:
    energy_threshold = 300

    def __init__(self, outcome):
        self.outcome = outcome           # str | dict | Exception, or a callable returning one
        self.prompts = []

    def listen(self, source, timeout=None, phrase_time_limit=None):
        o = self.outcome() if callable(self.outcome) else self.outcome
        if isinstance(o, BaseException):
            raise o
        return o

    def recognize_whisper(self, audio, model="tiny", initial_prompt="", show_dict=False, language=None):
        self.prompts.append(initial_prompt)
        return audio


def make(outcome, prompt=PROMPT, sync=True):
    q = queue.Queue()
    logs = []
    rec = FakeRecognizer(outcome)
    v = vi.VoiceInput(FakeSR, rec, q, get_prompt=lambda: prompt, play_cue=lambda: None,
                      stop_speech=lambda: None, log=logs.append,
                      start_thread=(lambda fn: fn()) if sync else None)
    return v, q, logs, rec


class Echo(unittest.TestCase):
    def test_prompt_echo_detected(self):
        for text in (OLD_PROMPT, PROMPT, "Sneha, water bottle, phone, laptop.", "Water bottle, medicine bottle.",
                     "sneha mummy water bottle", OLD_PROMPT + " Thank you."):
            self.assertTrue(vi.is_prompt_echo(text, OLD_PROMPT), text)

    def test_real_questions_are_not_echo(self):
        for text in ("This is Sneha", "this is my water bottle", "Where is my water bottle?",
                     "Is my phone near the laptop?", "Forget my medicine bottle", "forget Sneha", "Rename phone to laptop", "Rename my bottle to medicine bottle",
                     "What do you see?", "Find my phone", "Where is Sneha", "Who was here today?",
                     "what changed since I left", "I'm leaving"):
            self.assertFalse(vi.is_prompt_echo(text, OLD_PROMPT), text)

    def test_old_echo_would_have_been_a_forget_command(self):
        # The bug: this is what an echo used to turn into
        self.assertEqual(answers.classify_intent(OLD_PROMPT.lower()), "FORGET")

    def test_no_speech(self):
        self.assertTrue(vi.is_no_speech({"text": " you", "segments": [{"no_speech_prob": 0.9, "avg_logprob": -1.5}]}))
        self.assertFalse(vi.is_no_speech({"text": "this is sneha",
                                          "segments": [{"no_speech_prob": 0.1, "avg_logprob": -0.3}]}))
        self.assertFalse(vi.is_no_speech({"text": "x", "segments": [{"no_speech_prob": 0.9, "avg_logprob": -0.2}]}))
        self.assertTrue(vi.is_no_speech({"text": "", "segments": []}))
        self.assertFalse(vi.is_no_speech("plain text"))


class Listener(unittest.TestCase):
    def test_this_is_sneha_reaches_the_queue_and_is_a_learn(self):
        v, q, logs, rec = make({"text": " This is Sneha.", "segments": [{"no_speech_prob": 0.02, "avg_logprob": -0.2}]})
        self.assertTrue(v.press("window"))
        kind, text = q.get_nowait()
        self.assertEqual((kind, text), ("text", "This is Sneha."))
        self.assertEqual(answers.classify_intent(text.lower()), "LEARN")
        self.assertFalse(v.is_listening)
        self.assertEqual(rec.prompts, [PROMPT])

    def test_plain_string_result(self):
        v, q, _, _ = make("This is Sneha")
        v.press()
        self.assertEqual(q.get_nowait(), ("text", "This is Sneha"))

    def test_echo_never_reaches_the_answer_code(self):
        v, q, logs, _ = make({"text": OLD_PROMPT, "segments": [{"no_speech_prob": 0.3, "avg_logprob": -0.6}]})
        v.press()
        self.assertEqual(q.get_nowait()[0], "error")
        self.assertTrue(any("hint words" in l for l in logs))

    def test_timeout_exception_and_junk_all_reset(self):
        for outcome in (FakeSR.WaitTimeoutError(), OSError("PortAudio: Internal error -9986"),
                        {"text": "Thank you.", "segments": [{"no_speech_prob": 0.2, "avg_logprob": -0.5}]}):
            v, q, logs, _ = make(outcome)
            v.press()
            self.assertEqual(q.get_nowait()[0], "error")
            self.assertFalse(v.is_listening)
            self.assertEqual(v.stage, "idle")
            self.assertEqual(FakeSR.Microphone.open_now, 0)

    def test_press_while_busy_is_reported_not_silent(self):
        gate = threading.Event()

        def slow():
            gate.wait(2)
            return "what do you see"
        v, q, logs, _ = make(slow, sync=False)
        self.assertTrue(v.press("window"))
        time.sleep(0.05)
        self.assertFalse(v.press("terminal"))
        self.assertTrue(any("ignored" in l for l in logs))
        gate.set()
        self.assertEqual(q.get(timeout=2), ("text", "what do you see"))
        for _ in range(100):
            if not v.is_listening:
                break
            time.sleep(0.01)
        self.assertFalse(v.is_listening)
        self.assertTrue(v.press("window"))          # works again afterwards

    def test_thread_start_failure_does_not_lock_voice(self):
        q = queue.Queue()
        def boom(fn):
            raise RuntimeError("can't start new thread")
        v = vi.VoiceInput(FakeSR, FakeRecognizer("x"), q, get_prompt=lambda: "", play_cue=lambda: None,
                          stop_speech=lambda: None, log=lambda *a: None, start_thread=boom)
        self.assertFalse(v.press())
        self.assertFalse(v.is_listening)
        self.assertEqual(q.get_nowait()[0], "error")

    def test_mic_choice(self):
        self.assertEqual(vi.find_mic(FakeSR, "macbook")[0], 0)
        self.assertEqual(vi.find_mic(FakeSR, None), (None, None))
        idx, note = vi.find_mic(FakeSR, "usb")
        self.assertIsNone(idx)
        self.assertIn("Camo Microphone", note)


if __name__ == "__main__":
    unittest.main()


class SlowMicOnMac(unittest.TestCase):
    """Field log from the Mac: "mic open after 1.54s" and Whisper heard only
    'Sneha.'. PortAudio takes ~1 s to open. The person starts speaking when
    they hear the beep, so the beep must come AFTER the microphone is open."""

    def run_press(self, module):
        clock = {"t": 0.0, "beep": None, "mic_open": None}
        speech_len = 1.24                           # "This is Sneha"

        class Mic:
            def __init__(self, device_index=None):
                pass

            def __enter__(self):
                clock["t"] += 1.0                     # PortAudio start-up on the Mac
                clock["mic_open"] = clock["t"]
                return self

            def __exit__(self, *a):
                pass

        class SR:
            WaitTimeoutError = FakeSR.WaitTimeoutError
            Microphone = Mic

        class Rec:
            energy_threshold = 300

            def listen(self, source, timeout=None, phrase_time_limit=None):
                speech_start = clock["beep"]          # the user talks right after the beep
                missed = max(0.0, clock["mic_open"] - speech_start)
                return missed

            def recognize_whisper(self, missed, **kw):
                # what Whisper gets depends on how much of the sentence was recorded
                text = " This is Sneha." if missed < 0.1 else (" Sneha." if missed < speech_len else "")
                return {"text": text, "segments": [{"no_speech_prob": 0.05, "avg_logprob": -0.3}]}

        def beep():
            clock["t"] += 0.3
            clock["beep"] = clock["t"]

        q = queue.Queue()
        v = module.VoiceInput(SR, Rec(), q, get_prompt=lambda: PROMPT, play_cue=beep, stop_speech=lambda: None,
                              log=lambda *a: None, clock=lambda: clock["t"], start_thread=lambda fn: fn())
        v.press("window")
        return q.get_nowait()

    def test_whole_sentence_is_recorded_when_the_mic_is_slow(self):
        kind, text = self.run_press(vi)
        self.assertEqual(kind, "text")
        self.assertEqual(answers.classify_intent(text.lower()), "LEARN", text)
        self.assertEqual(answers.extract_name(text.lower()).strip(" .!?"), "sneha")
