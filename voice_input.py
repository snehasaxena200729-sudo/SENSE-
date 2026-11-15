"""
voice_input.py - SPACE -> beep -> record -> Whisper -> text for the main thread.

This is the listener that used to live inside sense_master.py
(listen_in_background / trigger_voice), moved here unchanged in behaviour so it
can be tested 10,000 times without a microphone. Same design as before:

    main thread                      listener thread (at most ONE, ever)
    -----------                      ---------------
    SPACE -> press()  --starts-->    beep, open mic, record until silence,
                                     close mic, Whisper (tiny, offline)
    process queue  <--("text", ..)-- put the transcript on the queue
    answer it (all SENSE state is touched only by the main thread)

What changed (field report: "SPACE does not capture my voice after a restart")
1. Every SPACE press is now logged with where it came from, and a press that
   arrives while the previous one is still recording or transcribing is
   reported ("busy") instead of being dropped silently.
2. Whisper hallucination guard. Whisper is given a hint prompt (known names,
   "water bottle", ...). On silence or a mumble the tiny model can return the
   PROMPT ITSELF as the transcript, e.g. "Sneha, water bottle, medicine bottle,
   phone, laptop, rename, forget." - which contains "forget" and was then
   executed as a FORGET command that deleted a person. The prompt only
   contains names AFTER a restart (they are loaded from SQLite), which is why
   this bites after restarting. Echoes of the prompt, and segments Whisper
   itself marks as "no speech", are now treated as "I didn't hear you".
3. The microphone can be chosen by name (--mic). Camo Studio and other apps can
   add virtual microphones; the one in use is printed at start-up.
"""
import os
import re
import subprocess
import threading
import time

EARCON = "/System/Library/Sounds/Tink.aiff"   # short macOS sound: "speak now"
WHISPER_JUNK = {"", "listening", "you", "thank you", "thanks for watching", "bye"}
LISTEN_START_TIMEOUT_S = 5.0     # time to START speaking
LISTEN_MAX_PHRASE_S = 5.0        # longest question
# Whisper's own "nothing was said" rule (openai-whisper transcribe.py uses the same pair)
NO_SPEECH_PROB = 0.6
LOW_LOGPROB = -1.0


def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def prompt_items(prompt):
    return [p.strip().lower() for p in re.split(r"[,.]", prompt or "") if p.strip()]


def is_prompt_echo(text, prompt):
    """True when the 'transcript' is just (part of) the hint prompt repeated back.

    Whisper copies the prompt's style: a comma-separated list of hint items and
    nothing else. Real commands mix hint words with other words ("forget MY
    bottle", "rename phone TO laptop"), so they are not treated as echoes:
        "Sneha, water bottle, phone, laptop."  -> echo
        "sneha mummy water bottle"             -> echo (3+ items, nothing else)
        "Forget my medicine bottle"            -> real command
    """
    items = prompt_items(prompt)
    if not items:
        return False
    t = " ".join(_words(text))
    hits = sum(1 for it in items if re.search(r"\b" + re.escape(" ".join(_words(it))) + r"\b", t))
    hint_words = {w for it in items for w in _words(it)}
    other = len([w for w in _words(text) if w not in hint_words])
    if hits >= 4 and other <= 2:
        return True
    if hits >= 3 and other == 0:
        return True
    return hits >= 2 and other == 0 and "," in text


def is_no_speech(result):
    """Whisper dict result -> True if every segment says 'no speech here'."""
    if not isinstance(result, dict):
        return False
    segs = result.get("segments") or []
    if not segs:
        return not (result.get("text") or "").strip()
    return all(s.get("no_speech_prob", 0.0) > NO_SPEECH_PROB and s.get("avg_logprob", 0.0) < LOW_LOGPROB
               for s in segs)


def audio_to_float(audio):
    """SpeechRecognition AudioData -> 16 kHz mono float32 in [-1, 1], what Whisper
    expects (same as SpeechRecognition's own Whisper code, without soundfile)."""
    import numpy as np
    raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def flush_input(source):
    """Discard audio already waiting in the microphone buffer (the beep).
    Returns the seconds discarded (0 if the stream can't tell)."""
    try:
        n = source.stream.pyaudio_stream.get_read_available()
    except Exception:  # noqa: BLE001  (not a PyAudio stream, e.g. tests)
        return 0.0
    if n > 0:
        source.stream.read(n)
    return n / float(getattr(source, "SAMPLE_RATE", 16000) or 16000)


def audio_seconds(audio):
    try:
        return len(audio.get_raw_data()) / float(audio.sample_rate * audio.sample_width)
    except Exception:  # noqa: BLE001
        return 0.0


def find_mic(sr_module, wanted):
    """Index of the first input device whose name contains `wanted` (case-insensitive)."""
    if not wanted:
        return None, None
    try:
        names = sr_module.Microphone.list_microphone_names()
    except Exception as e:  # noqa: BLE001
        return None, f"could not list microphones ({e})"
    for i, n in enumerate(names):
        if n and wanted.lower() in n.lower():
            return i, n
    return None, f"no microphone matching '{wanted}' (have: {', '.join(n for n in names if n)})"


def default_mic_name():
    """Name of the default input device (PyAudio), for the start-up log."""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        try:
            return pa.get_default_input_device_info().get("name")
        finally:
            pa.terminate()
    except Exception:  # noqa: BLE001
        return None


class VoiceInput:
    def __init__(self, sr_module, recognizer, out_queue, get_prompt, play_cue=None, stop_speech=None,
                 log=print, mic_index=None, model="tiny", clock=time.time, start_thread=None):
        self.sr = sr_module
        self.recognizer = recognizer
        self.queue = out_queue
        self.get_prompt = get_prompt
        self.play_cue = play_cue or self._default_cue
        self.stop_speech = stop_speech or (lambda: subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL))
        self.log = log
        self.mic_index = mic_index
        self.model = model
        self.clock = clock
        self.start_thread = start_thread or (lambda fn: threading.Thread(target=fn, daemon=True).start())
        self.is_listening = False
        self.stage = "idle"          # idle / cue / recording / transcribing (for the log)
        self.busy_since = None
        self.presses = 0
        self.ignored = 0
        self._model = None

    @staticmethod
    def _default_cue():
        if os.path.exists(EARCON):
            subprocess.run(["afplay", EARCON], timeout=2, stderr=subprocess.DEVNULL)

    def press(self, where="window"):
        """SPACE was pressed. Starts the listener unless one is already running.
        Returns True if a new recording started."""
        self.presses += 1
        if self.is_listening:
            self.ignored += 1
            self.log(f"⌨️  SPACE ({where}) ignored: still {self.stage} "
                     f"({self.clock() - self.busy_since:.1f}s). Wait for the answer, then press again.")
            return False
        # Set the flag BEFORE starting the thread, so a second key press can
        # never start a second listener (keeps SENSE at 2 threads max).
        self.is_listening = True
        self.stage = "starting"
        self.busy_since = self.clock()
        self.log(f"⌨️  SPACE ({where}) -> listening")
        try:
            self.start_thread(self._listen)
        except Exception as e:  # noqa: BLE001  (e.g. thread could not start)
            self.is_listening, self.stage = False, "idle"
            self.log(f"Ear note: could not start the listener: {e}")
            self.queue.put(("error", "Sorry, I could not start listening."))
            return False
        return True

    def preload(self):
        """Load the Whisper model ONCE, at start-up (main thread).

        SpeechRecognition 3.11+ calls whisper.load_model() inside EVERY
        recognize_whisper() call, i.e. it re-reads the model from disk for each
        question. Loading it once removes that delay, and a missing model (the
        ~/.cache/whisper download was never made or was deleted, and SENSE is
        offline) is reported clearly at start-up instead of as "Sorry, I had
        trouble hearing that" on every press. Returns True if loaded."""
        try:
            import whisper
            t = self.clock()
            self._model = whisper.load_model(self.model)
            self.log(f"🎙️ Whisper '{self.model}' loaded in {self.clock() - t:.1f}s (offline).")
            return True
        except Exception as e:  # noqa: BLE001
            self._model = None
            self.log(f"🎙️ Whisper '{self.model}' could not be loaded: {type(e).__name__}: {e}. "
                     "Voice questions will not work until it is downloaded once (needs internet).")
            return False

    def _transcribe(self, audio, prompt):
        """Whisper, offline, English. Returns Whisper's dict (text + segments)."""
        model = getattr(self, "_model", None)
        if model is not None:
            return model.transcribe(audio_to_float(audio), initial_prompt=prompt, language="en", fp16=False)
        try:
            return self.recognizer.recognize_whisper(audio, model=self.model, initial_prompt=prompt,
                                                     show_dict=True, language="english")
        except TypeError:
            return self.recognizer.recognize_whisper(audio, model=self.model, initial_prompt=prompt)

    def _listen(self):
        t0 = self.clock()
        t_mic = t0
        try:
            self.stage = "opening mic"
            self.stop_speech()
            mic_kwargs = {} if self.mic_index is None else {"device_index": self.mic_index}
            # Field log (krish, Mac): "mic open after 1.54s". Starting PortAudio
            # takes ~1 s on the Mac. The beep used to play BEFORE that, so the
            # first words ("This is") were spoken into a closed microphone and
            # Whisper only got "Sneha". Now: open the mic first, THEN beep, so
            # the beep really means "recording now".
            with self.sr.Microphone(**mic_kwargs) as source:
                t_mic = self.clock()
                self.log(f"\n🎙️ Mic opened after {t_mic - t0:.2f}s")
                self.stage = "cue"
                self.play_cue()
                dropped = flush_input(source)          # the beep itself is not speech
                t_cue = self.clock()
                self.stage = "recording"
                self.log(f"🎙️ Beep done (+{t_cue - t_mic:.2f}s, {dropped:.2f}s of beep audio discarded). "
                         f"Listening now... energy threshold "
                         f"{getattr(self.recognizer, 'energy_threshold', 0):.0f}")
                audio = self.recognizer.listen(source, timeout=LISTEN_START_TIMEOUT_S,
                                               phrase_time_limit=LISTEN_MAX_PHRASE_S)
                t_mic = t_cue
            t_rec = self.clock()        # microphone is released before the (slower) transcription
            self.stage = "transcribing"
            prompt = self.get_prompt()
            result = self._transcribe(audio, prompt)
            text = (result.get("text", "") if isinstance(result, dict) else str(result)).strip()
            t_done = self.clock()
            self.log(f"👂 Heard: '{text}'  ⏱ audio {audio_seconds(audio):.1f}s, listened {t_rec - t_mic:.1f}s, "
                     f"Whisper {t_done - t_rec:.1f}s, total {t_done - t0:.1f}s since space")
            if is_no_speech(result):
                self.log("👂 Whisper marked this as no speech: ignored.")
                self.queue.put(("error", "I didn't hear a question. Press space and speak after the beep."))
            elif is_prompt_echo(text, prompt):
                self.log("👂 That is Whisper repeating its hint words (nothing clear was said): ignored.")
                self.queue.put(("error", "I didn't hear a question. Press space and speak after the beep."))
            elif text.lower().strip(" .!?,") in WHISPER_JUNK:
                self.queue.put(("error", "I didn't hear a question. Press space and speak after the beep."))
            else:
                self.queue.put(("text", text))   # do NOT answer here: the main thread does
        except self.sr.WaitTimeoutError:
            self.log("👂 Nothing heard within the time limit.")
            self.queue.put(("error", "I didn't hear anything. Press space and speak after the beep."))
        except Exception as e:  # noqa: BLE001
            self.log(f"Ear note: {type(e).__name__}: {e}")
            self.queue.put(("error", "Sorry, I had trouble hearing that."))
        finally:
            self.stage = "idle"
            self.is_listening = False
