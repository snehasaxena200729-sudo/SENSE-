"""key_input.py: SPACE / C / Q read from a real pseudo-terminal (like the VS Code
terminal), plus the cv2.waitKey value mapping."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import key_input  # noqa: E402

try:
    import pty
    import termios
except ImportError:          # Windows
    pty = None


@unittest.skipIf(pty is None, "needs a POSIX terminal")
class TerminalKeysTest(unittest.TestCase):
    def setUp(self):
        self.master, self.slave = pty.openpty()
        self.stream = os.fdopen(self.slave, "r")
        self.before = termios.tcgetattr(self.slave)

    def tearDown(self):
        self.stream.close()
        os.close(self.master)

    def test_keys_arrive_without_enter_and_terminal_is_restored(self):
        keys = key_input.TerminalKeys(stream=self.stream)
        self.assertTrue(keys.enabled)
        self.assertEqual(keys.poll(), [])
        os.write(self.master, b" ")          # no Enter: cbreak mode delivers it at once
        self.assertEqual(keys.poll(), [ord(" ")])
        os.write(self.master, b"Cq ")
        self.assertEqual(keys.poll(), [ord("c"), ord("q"), ord(" ")])
        keys.restore()
        self.assertEqual(termios.tcgetattr(self.slave), self.before)
        self.assertEqual(keys.poll(), [])

    def test_disabled(self):
        keys = key_input.TerminalKeys(stream=self.stream, enabled=False)
        os.write(self.master, b" ")
        self.assertEqual(keys.poll(), [])
        self.assertEqual(termios.tcgetattr(self.slave), self.before)


class WindowKey(unittest.TestCase):
    def test_mapping(self):
        self.assertIsNone(key_input.window_key(-1))
        self.assertIsNone(key_input.window_key(255))
        self.assertEqual(key_input.window_key(32), ord(" "))
        self.assertEqual(key_input.window_key(ord("Q")), ord("q"))
        self.assertEqual(key_input.window_key(0x100000 | ord("c")), ord("c"))   # macOS modifier bits

    def test_not_a_tty(self):
        import io
        self.assertFalse(key_input.TerminalKeys(stream=io.StringIO()).enabled)


if __name__ == "__main__":
    unittest.main()
