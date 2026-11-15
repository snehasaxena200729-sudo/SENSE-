"""The REAL SpeechRecognition recorder on real audio files.

recognizer.listen() is SpeechRecognition's own code (energy threshold, start
timeout, pause detection); only the microphone is replaced by a WAV file:
    tests/audio/this_is_sneha.wav   0.8 s room hiss, "This is Sneha" (espeak), 2.5 s hiss
    tests/audio/silence.wav         7 s room hiss, nobody speaks
Whisper itself runs only if its 'tiny' model is already downloaded
(~/.cache/whisper/tiny.pt, i.e. on the Mac where SENSE runs); otherwise that
part is skipped and a stand-in returns the text.
"""
import os
import queue
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import answers  # noqa: E402
import voice_input as vi  # noqa: E402

try:
    import speech_recognition as sr
except ImportError:
    sr = None

AUDIO = os.path.join(ROOT, "tests", "audio")
WHISPER_TINY = os.path.join(os.path.expanduser("~"), ".cache", "whisper", "tiny.pt")
PROMPT = "Sneha, water bottle, medicine bottle, phone, laptop, rename."


def sense_recognizer(cls):
    r = cls()
    r.pause_threshold = 1.0      # same settings as sense_master.py
    r.energy_threshold = 300
    return r


@unittest.skipIf(sr is None, "SpeechRecognition not installed")
class RealRecorder(unittest.TestCase):
    def run_file(self, name, real_whisper=False):
        heard = {}

        class Rec(sr.Recognizer):
            def recognize_whisper(self, audio, **kw):
                heard["seconds"] = len(audio.get_raw_data()) / (audio.sample_rate * audio.sample_width)
                heard["samples"] = vi.audio_to_float(audio)
                return {"text": " This is Sneha.", "segments": [{"no_speech_prob": 0.05, "avg_logprob": -0.3}]}

        shim = types.SimpleNamespace(Microphone=lambda **k: sr.AudioFile(os.path.join(AUDIO, name)),
                                     WaitTimeoutError=sr.WaitTimeoutError)
        q, logs = queue.Queue(), []
        v = vi.VoiceInput(shim, sense_recognizer(Rec), q, get_prompt=lambda: PROMPT, play_cue=lambda: None,
                          stop_speech=lambda: None, log=logs.append, start_thread=lambda fn: fn())
        if real_whisper:
            self.assertTrue(v.preload())
        v.press("window")
        return q.get_nowait(), heard, logs, v

    def test_speech_is_recorded_and_recording_stops_at_the_pause(self):
        (kind, text), heard, logs, v = self.run_file("this_is_sneha.wav")
        self.assertEqual(kind, "text")
        # the phrase is ~1.24 s; recording must stop about pause_threshold (1 s) after it,
        # long before the 2.5 s of trailing hiss runs out or the 5 s phrase limit
        self.assertGreater(heard["seconds"], 1.2)
        self.assertLess(heard["seconds"], 3.4)
        self.assertEqual(heard["samples"].dtype.name, "float32")
        self.assertLessEqual(float(abs(heard["samples"]).max()), 1.0)
        self.assertFalse(v.is_listening)

    def test_silence_times_out_cleanly(self):
        (kind, text), heard, logs, v = self.run_file("silence.wav")
        self.assertEqual(kind, "error")
        self.assertIn("didn't hear anything", text)
        self.assertNotIn("seconds", heard)       # Whisper never ran
        self.assertFalse(v.is_listening)

    @unittest.skipUnless(os.path.exists(WHISPER_TINY), "Whisper tiny model not downloaded on this machine")
    def test_real_whisper_this_is_sneha(self):
        shim = types.SimpleNamespace(Microphone=lambda **k: sr.AudioFile(os.path.join(AUDIO, "this_is_sneha.wav")),
                                     WaitTimeoutError=sr.WaitTimeoutError)
        q = queue.Queue()
        v = vi.VoiceInput(shim, sense_recognizer(sr.Recognizer), q, get_prompt=lambda: PROMPT,
                          play_cue=lambda: None, stop_speech=lambda: None, start_thread=lambda fn: fn())
        self.assertTrue(v.preload())
        v.press("window")
        kind, text = q.get_nowait()
        print(f"\nReal Whisper heard: {text!r}")
        self.assertEqual(kind, "text")
        self.assertEqual(answers.classify_intent(text.lower()), "LEARN")

    @unittest.skipUnless(os.path.exists(WHISPER_TINY), "Whisper tiny model not downloaded on this machine")
    def test_real_whisper_on_hiss_never_becomes_a_command(self):
        # 7 s of hiss: SENSE times out before Whisper; feed the hiss straight to Whisper too
        import whisper
        model = whisper.load_model("tiny")
        audio = sr.AudioFile(os.path.join(AUDIO, "silence.wav"))
        with audio as src:
            data = sense_recognizer(sr.Recognizer).record(src)
        result = model.transcribe(vi.audio_to_float(data), initial_prompt=PROMPT, language="en", fp16=False)
        text = result["text"].strip()
        print(f"\nReal Whisper on hiss: {text!r}")
        dropped = (vi.is_no_speech(result) or vi.is_prompt_echo(text, PROMPT)
                   or text.lower().strip(" .!?,") in vi.WHISPER_JUNK)
        self.assertTrue(dropped or answers.classify_intent(text.lower()) not in ("FORGET", "RENAME", "LEARN"),
                        text)


if __name__ == "__main__":
    unittest.main()
