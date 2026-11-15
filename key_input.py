"""
key_input.py - SENSE's keys (SPACE = talk, Q = quit) from the
OpenCV window AND from the terminal, without an extra thread.

Root cause this fixes
---------------------
SENSE only read keys with cv2.waitKey(), and on macOS waitKey() only sees a key
press while the OpenCV window is the focused ("key") window. When SENSE is
started from the VS Code / Terminal window, the video window often opens
BEHIND it or without focus, so SPACE goes to the terminal (it just prints a
space) and SENSE never hears it. Nothing in the voice code runs at all, which
looks exactly like "voice input is broken after a restart".

How the terminal is read without a thread
-----------------------------------------
The terminal is put in "cbreak" mode (each key is delivered at once instead of
after Enter; Ctrl+C still works) and the main loop asks select() "is a key
waiting?" with a timeout of 0 once per frame. That costs microseconds and
keeps SENSE at its two-thread budget. The terminal mode is restored on exit
(atexit), even after a crash.

Trade-off: a global keyboard hook (pynput) would catch SPACE whichever app is
focused, but it needs an extra thread, a new dependency and macOS
Accessibility permission, and it would also fire while you type in other apps.
"""
import atexit
import os
import sys

KEY_NAMES = {ord(" "): "SPACE", ord("q"): "Q"}


class TerminalKeys:
    def __init__(self, stream=None, enabled=True):
        self.stream = stream if stream is not None else sys.stdin
        self.fd = None
        self.old = None
        self.enabled = False
        if not enabled:
            return
        try:
            import termios
            import tty
            if self.stream is None or not self.stream.isatty():
                return
            self.fd = self.stream.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.enabled = True
            atexit.register(self.restore)
        except Exception:           # Windows, IDE consoles without a tty, ...
            self.enabled = False

    def poll(self):
        """Keys pressed in the terminal since the last call, lower-cased (list of ord)."""
        if not self.enabled:
            return []
        import select
        keys = []
        try:
            while select.select([self.fd], [], [], 0)[0]:
                data = os.read(self.fd, 64)
                if not data:
                    break
                keys.extend(ord(ch.lower()) for ch in data.decode(errors="ignore"))
        except (OSError, ValueError):
            return keys
        return keys

    def restore(self):
        if self.enabled and self.old is not None:
            try:
                import termios
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            except Exception:
                pass
            self.enabled = False


def window_key(raw):
    """cv2.waitKey() value -> lower-case ord, or None for 'no key'."""
    if raw is None or raw < 0:
        return None
    k = raw & 0xFF
    if k == 255:
        return None
    return ord(chr(k).lower()) if 65 <= k <= 90 else k
