"""Regression: "this is Sneha" must not lose "this is" (mic opened after the beep)."""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import answers  # noqa: E402

SRC = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "sense_master.py"), encoding="utf-8").read()


class TestMicBeforeBeep(unittest.TestCase):
    def test_mic_opens_before_beep(self):
        body = SRC.split("def listen_in_background", 1)[1].split("\ndef ", 1)[0]
        self.assertLess(body.index("sr.Microphone()"), body.index('["afplay"'),
                        "the beep must play only after the microphone is open")

    def test_beep_does_not_block_recording(self):
        body = SRC.split("def listen_in_background", 1)[1].split("\ndef ", 1)[0]
        self.assertNotIn('subprocess.run(["afplay"', body,
                         "a blocking beep leaves the open mic unread and drops the first words")

    def test_lone_name_gets_retry_hint(self):
        self.assertTrue(re.search(r"I only heard \{lone\}", SRC))



class TestClippedIntroduction(unittest.TestCase):
    def test_exact_transcripts_from_mac_log(self):
        # krish, 2026-11-16: said "This is Sneha", Whisper returned these
        for heard in ["Is Sneha.", "Listen, is Sneha.", "is sneha.", "listen, is sneha."]:
            self.assertEqual(answers.classify_intent(heard), "WHERE")   # why it failed before
            self.assertEqual(answers.parse_introduction(heard), "sneha", heard)

    def test_other_clipped_forms(self):
        for heard in ["It is Sneha.", "Dis is Sneha", "This's Sneha.", "So, this is Sneha"]:
            self.assertEqual(answers.parse_introduction(heard), "sneha", heard)

    def test_real_questions_are_not_introductions(self):
        for heard in ["Where is Sneha?", "Is Sneha here?", "The phone is red.", "Who is Sneha?",
                      "Sneha.", "Where was my bottle", "is it", "my bottle is here",
                      "what is this"]:
            self.assertIsNone(answers.parse_introduction(heard), heard)

    def test_hooked_into_answer_voice_query(self):
        self.assertIn("answers.parse_introduction(user_text)", SRC)


if __name__ == "__main__":
    unittest.main()
