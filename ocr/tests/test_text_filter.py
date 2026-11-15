import unittest

from ocr.config import OCRConfig
from ocr.ocr_engine import parse_tsv
from ocr.tests.helpers import words
from ocr.text_filter import RepeatFilter, extract_text, is_junk, similarity, truncate_words

TSV = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
       "1\t1\t0\t0\t0\t0\t0\t0\t640\t480\t-1\t\n"
       "5\t1\t1\t1\t1\t1\t10\t10\t50\t20\t96.5\tTAKE\n"
       "5\t1\t1\t1\t1\t2\t70\t10\t20\t20\t91.0\t2\n"
       "5\t1\t1\t1\t2\t1\t10\t40\t50\t20\t88.0\tDAILY\n"
       "5\t1\t2\t1\t1\t1\t300\t300\t20\t20\t35.0\tqx\n"
       "5\t1\t2\t1\t1\t2\t330\t300\t20\t20\t90.0\t \n")


class ParseTests(unittest.TestCase):
    def test_parse_tsv_keeps_only_words(self):
        ws = parse_tsv(TSV)
        self.assertEqual([w.text for w in ws], ["TAKE", "2", "DAILY", "qx"])
        self.assertEqual(ws[0].conf, 96.5)

    def test_parse_garbage_is_empty_not_crash(self):
        self.assertEqual(parse_tsv(""), [])
        self.assertEqual(parse_tsv("not a tsv\n1\t2"), [])

    def test_lines_and_confidence(self):
        self.assertEqual(extract_text(parse_tsv(TSV), OCRConfig()), "TAKE 2. DAILY")


class FilterTests(unittest.TestCase):
    cfg = OCRConfig()

    def test_low_confidence_dropped(self):
        self.assertEqual(extract_text(words(("EXIT", 95), ("MAZE", 20)), self.cfg), "EXIT")

    def test_junk_and_lonely_short_tokens_dropped(self):
        self.assertTrue(is_junk("|"))
        self.assertTrue(is_junk("é"))
        self.assertEqual(extract_text(words(("ay", 90, 1), ("ie", 90, 2), ("4", 90, 3)), self.cfg), "")

    def test_short_words_kept_next_to_real_word(self):
        self.assertEqual(extract_text(words(("Take", 90), ("1", 90), ("a", 90), ("day", 90)), self.cfg),
                         "Take 1 a day")

    def test_too_little_text_is_nothing(self):
        self.assertEqual(extract_text([], self.cfg), "")

    def test_truncation(self):
        long = " ".join(f"word{i}" for i in range(100))
        out = truncate_words(long, 40)
        self.assertEqual(len(out.split()), 40 + 4)
        self.assertTrue(out.endswith("and more text"))


class RepeatTests(unittest.TestCase):
    def test_similar_text_is_repeat_until_cooldown(self):
        f = RepeatFilter(0.85, 30)
        f.remember("EMERGENCY EXIT", 0)
        self.assertTrue(f.is_repeat("EMERGENCY EXlT", 5))      # one misread letter
        self.assertFalse(f.is_repeat("WET FLOOR", 5))
        self.assertFalse(f.is_repeat("EMERGENCY EXIT", 31))    # cooldown over

    def test_similarity_ignores_case_and_punctuation(self):
        self.assertEqual(similarity("Exit.", "exit"), 1.0)


if __name__ == "__main__":
    unittest.main()
