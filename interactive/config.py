"""
interactive/config.py - every tunable number of the question-answering layer
in one place, with the reason for its value.
"""
import os

# --- What counts as "now" ---------------------------------------------------
FRESH_S = 1.2            # a YOLO frame older than this is not "what I see now".
                         # (YOLO runs every frame when steady, every 3rd while moving)
LIVE_MEMORY_S = 2.5      # same "visible right now" window the existing answers use

# --- Follow-up questions ("where is it?") -----------------------------------
SUBJECT_TTL_S = 60.0     # "it" refers to the last thing asked about for 60 s

# --- Colour ------------------------------------------------------------------
MIN_CROP_PX = 25         # same limit the existing colour signature uses
DARK_V = 45              # an object this dark (median V)...
DARK_SCENE_V = 50        # ...in a frame this dark is "too dark to tell", not "black"
EDGE_PX = 3              # a box this close to the frame edge is cut off
CORE_FRACTION = 0.6      # look at the middle 60% of the box: less background
AMBIGUOUS_SHARE = 0.55   # winning hue must hold 55% of the coloured pixels...
RUNNER_UP_SHARE = 0.30   # ...or a runner-up with 30% makes it "blue or green"

# --- Earlier / behind ----------------------------------------------------------
PAST_WINDOW_S = 1800     # "what was in front of me" looks back 30 minutes
DIRECTION_HALF_WIDTH = {"ahead": 35.0}   # +-35 degrees counts as "in front of you"

# --- Speaking ------------------------------------------------------------------
P1_HOLD_MAX_S = 4.0      # an answer waits at most this long for a safety alert to end
MAX_LIST = 4             # never list more than 4 things in one sentence

# --- Local AI model (Ollama, fully offline) -------------------------------------
LLM_ENABLED = os.environ.get("SENSE_LLM", "on").lower() not in ("0", "off", "no", "false")
LLM_MODEL = os.environ.get("SENSE_LLM_MODEL", "llama3.2:3b")
LLM_HOST = os.environ.get("SENSE_LLM_HOST", "http://127.0.0.1:11434")
LLM_TIMEOUT_S = float(os.environ.get("SENSE_LLM_TIMEOUT", "8"))
LLM_FILLER_AFTER_S = 1.5  # say "One moment." if the model has not answered by then
LLM_MAX_TOKENS = 70       # one or two short sentences
LLM_KEEP_ALIVE = "30m"    # keep the model in RAM between questions (fast answers)
