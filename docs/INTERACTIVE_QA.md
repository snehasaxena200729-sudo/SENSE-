# Asking SENSE questions (interactive/)

Press **space**, wait for `🎙️ Listening now...`, then ask a question. Everything runs on the Mac; nothing needs the internet.

| You ask | SENSE answers from | Example answer |
| --- | --- | --- |
| What colour is the bottle? / Is the cup red? | the bottle's picture in the last camera frame | "The bottle looks blue." / "I can only see part of the bottle. That part looks blue." |
| How many chairs are in front of me? / How many people are around me? | the objects tracked in the last frame, each counted once | "I can see two chairs: one on your left and one in front of you." |
| Is there anything on my right? / Is there a chair on my left? | the last frame and the door model | "On your right, I can see a cup, within arm's reach." |
| Is it safe to walk forward? / Any obstacles ahead? | the last frame | "Careful. In front of you I can see a chair, about 3 to 4 feet away. I can't promise the way is clear." (never says "safe") |
| Describe the object in front of me. / What am I holding? | the last frame + colour + distance | "A bottle, in front of you, about 2 to 3 feet away. It looks blue." |
| What was previously in front of me? / What is behind me? | memory (this run), turned to where you face now | "Earlier, in front of you from where you face now, I saw your phone 4 minutes ago." |
| What time is it? / What is today's date? | the Mac's clock | "It is 3:45 PM." / "Today is Monday, 16 November 2026." |
| What does this medicine label say? | the existing text reader (same as "read this") | "Reading. Hold it steady." |
| Where is it? / What colour is it? (follow-up, within 60 s) | the thing you asked about last | uses the existing "where is" answer |
| Anything else, e.g. "Can I sit somewhere?" | the local AI model, only if it is installed (below) | "I can see a chair in front of you." |

Questions SENSE already understood (where is my phone, how far is the door, find my bottle, what changed, who was here, this is Sneha, read this, ...) still go to the **same code as before**.

## How honest answers work

- **Now vs earlier.** "I can see" only means the last camera frame (under 1.2 s old). Memory answers always say when ("4 minutes ago").
- **Unclear colour.** Too small, cut off by the edge of the picture, too dark, or two colours: SENSE says so instead of guessing.
- **People.** Only people introduced by face are named. Everyone else is "a person".
- **Distances** come from the existing distance code, which already says "distance unclear" when it can't tell.
- **Safety first.** If a safety warning is being spoken, the answer waits until it ends (at most 4 s).

## Optional: the local AI model (offline, for free-form questions)

The rule-based answers above work without it. The model only handles questions the rules and the old code can't place, and its answer is fact-checked before it is spoken (any object, colour, number or name that SENSE did not see makes it say "I'm not sure" instead).

### One-time setup on the Mac (needs internet once, about 2 GB)

```bash
brew install ollama            # or download the app from https://ollama.com/download
ollama serve &                 # or just open the Ollama app; it runs in the menu bar
ollama pull llama3.2:3b        # the model, about 2 GB
ollama run llama3.2:3b "Say hello in five words."   # quick test, then Ctrl+D
```

After that, Wi-Fi can stay off. Ollama must be running (menu-bar icon) **before** you start SENSE. At start-up the terminal prints one of:

- `💬 Local AI model ready: llama3.2:3b (offline, via Ollama).`
- `💬 Local AI model off: Ollama is not running ...` (open the Ollama app, restart SENSE)
- `💬 Local AI model off: 'llama3.2:3b' is not downloaded ...` (run `ollama pull llama3.2:3b`)

### Settings (environment variables, optional)

| Variable | Default | Meaning |
| --- | --- | --- |
| `SENSE_LLM` | `on` | `off` = rules only (no model, no Ollama check) |
| `SENSE_LLM_MODEL` | `llama3.2:3b` | e.g. `llama3.2:1b` (faster, about 1.3 GB, less careful) |
| `SENSE_LLM_TIMEOUT` | `8` | seconds before SENSE gives up and says so |

Example: `SENSE_LLM=off python3 sense_master.py`.

### What it costs

Answers take about 1 to 3 seconds on an M-series Mac (SENSE says "One moment." after 1.5 s). While the model writes, the video can stutter slightly because the model shares the Mac's chip; detection and safety warnings keep running, because the model runs in its own process and SENSE never waits for it.

## Design notes (for the report)

- **Two paths.** `sense_master.py` (Path A) does not depend on `interactive/` (Path B). If the package fails to load, SENSE prints `💬 Interactive questions off` and runs exactly as before.
- **No new threads, camera reads or detectors.** The main loop hands Path B a read-only reference to each confirmed detection. The model runs through `curl` as a separate process, polled once per frame.
- **Why rules first and a model second.** Rules answer in under a millisecond and can't invent anything. A local model understands free wording but is slower and can hallucinate, so it only gets a short fact sheet and is fact-checked (`interactive/guard.py`).
- **Tests.** `tests/test_interactive.py` (unit tests, including a fake Ollama server) and `tools/stress_interactive.py` (10,000 seeded cases: new questions with known answers, existing commands that must stay untouched, follow-up chains, and grounded and hallucinated model answers).
