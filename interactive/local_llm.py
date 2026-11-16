"""
interactive/local_llm.py - a local language model through Ollama, OFFLINE.

Ollama (https://ollama.com) runs the model on the Mac itself and listens on
http://127.0.0.1:11434. Nothing leaves the computer. SENSE talks to it with
the `curl` program started as a separate PROCESS (subprocess.Popen):

  * no Python thread is added (the 2-thread rule holds), and
  * the main loop never waits: it starts the request, then checks once per
    frame with poll() whether curl has finished - the same pattern SENSE
    already uses for `say` and for Tesseract.

If Ollama is not installed, not running, or the model is not downloaded,
status() says so and SENSE uses the rule-based answers only.
"""
import json
import subprocess
import time

from . import config as C


class Job:
    def __init__(self, proc, started):
        self.proc = proc
        self.started = started

    def done(self):
        return self.proc.poll() is not None

    def result(self, raw=False):
        """(text or None, error or None). Call only when done().
        raw=True returns the whole JSON text (for /api/tags)."""
        try:
            out, err = self.proc.communicate(timeout=1)
        except Exception as e:      # pragma: no cover
            return None, f"read failed: {e}"
        if self.proc.returncode != 0:
            return None, f"curl exit {self.proc.returncode}" + (": timeout" if self.proc.returncode == 28 else "")
        if raw:
            return out.decode("utf-8", "replace"), None
        try:
            data = json.loads(out.decode("utf-8", "replace") or "{}")
        except ValueError:
            return None, "bad JSON from Ollama"
        if data.get("error"):
            return None, str(data["error"])
        return (data.get("response") or "").strip(), None

    def cancel(self):
        if self.proc.poll() is None:
            try:
                self.proc.kill()
                self.proc.communicate(timeout=1)
            except Exception:
                pass


class OllamaLLM:
    def __init__(self, model=C.LLM_MODEL, host=C.LLM_HOST, timeout_s=C.LLM_TIMEOUT_S,
                 popen=subprocess.Popen, clock=time.time, log=print):
        self.model, self.host, self.timeout_s = model, host.rstrip("/"), timeout_s
        self.popen, self.clock, self.log = popen, clock, log
        self.state = "checking"          # checking | ready | no_server | no_model | off
        self._probe = None
        self._warm = None

    # ------------------------------------------------------------ plumbing
    def _curl(self, path, body=None, max_time=None):
        cmd = ["curl", "-s", "--noproxy", "*", "-m", str(max_time or self.timeout_s),
               f"{self.host}{path}"]
        if body is not None:
            cmd[1:1] = ["-X", "POST", "-H", "Content-Type: application/json", "--data-binary", "@-"]
        proc = self.popen(cmd, stdin=subprocess.PIPE if body is not None else subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if body is not None:
            proc.stdin.write(json.dumps(body).encode("utf-8"))
            proc.stdin.close()
        return Job(proc, self.clock())

    # ------------------------------------------------------------ start-up
    def start(self):
        """Non-blocking: ask Ollama which models it has. Finished in tick()."""
        try:
            self._probe = self._curl("/api/tags", max_time=3)
        except FileNotFoundError:
            self.state = "off"
            self.log("💬 Local AI model off: the curl program was not found.")

    def tick(self):
        """Called every frame (cheap): finishes the start-up probe and warm-up."""
        if self._probe is not None and self._probe.done():
            proc, self._probe = self._probe, None
            text, err = proc.result(raw=True)
            if err is not None or not text:
                err = err or "no reply"
                self.state = "no_server"
                self.log(f"💬 Local AI model off: Ollama is not running ({err}). "
                         f"Start the Ollama app, then restart SENSE. Rule-based answers still work.")
                return
            try:
                names = [m.get("name", "") for m in json.loads(text or "{}").get("models", [])]
            except Exception:
                names = []
            wanted = self.model if ":" in self.model else self.model + ":latest"
            if not any(n == wanted or n == self.model for n in names):
                self.state = "no_model"
                self.log(f"💬 Local AI model off: '{self.model}' is not downloaded. "
                         f"Run once with internet: ollama pull {self.model}")
                return
            self.state = "ready"
            self.log(f"💬 Local AI model ready: {self.model} (offline, via Ollama).")
            # load the model into memory now, so the first question is fast
            self._warm = self._curl("/api/generate", {"model": self.model, "prompt": "",
                                                       "keep_alive": C.LLM_KEEP_ALIVE}, max_time=60)
        if self._warm is not None and self._warm.done():
            self._warm.result()
            self._warm = None

    @property
    def ready(self):
        return self.state == "ready"

    def ask(self, system, prompt):
        return self._curl("/api/generate", {
            "model": self.model, "system": system, "prompt": prompt, "stream": False,
            "keep_alive": C.LLM_KEEP_ALIVE,
            "options": {"temperature": 0, "num_predict": C.LLM_MAX_TOKENS, "num_ctx": 2048},
        })

    def close(self):
        for j in (self._probe, self._warm):
            if j is not None:
                j.cancel()
