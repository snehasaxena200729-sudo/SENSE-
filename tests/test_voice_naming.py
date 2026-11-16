"""Regression: "this is Sneha" must not lose "this is" (mic opened after the beep)."""
import os
import re
import unittest

SRC = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "sense_master.py"), encoding="utf-8").read()


class TestMicBeforeBeep(unittest.TestCase):
    def test_mic_opens_before_beep(self):
        body = SRC.split("def listen_in_background", 1)[1].split("\ndef ", 1)[0]
        self.assertLess(body.index("sr.Microphone()"), body.index('["afplay"'),
                        "the beep must play only after the microphone is open")

    def test_lone_name_gets_retry_hint(self):
        self.assertTrue(re.search(r"I only heard \{lone\}", SRC))


if __name__ == "__main__":
    unittest.main()
