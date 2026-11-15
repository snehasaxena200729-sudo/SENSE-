"""
ocr/config.py - every OCR setting in one place.

Nothing else in ocr/ hard-codes a number: change behaviour here, or pass your
own OCRConfig to OCRController(config=...) in a test.
"""

from dataclasses import dataclass, field


@dataclass
class OCRConfig:
    # --- Tesseract program ------------------------------------------------
    # Tesseract is a system program (macOS: `brew install tesseract`), not a pip
    # package. Apps started from VS Code often do not have Homebrew on PATH, so
    # the usual install folders are tried too.
    tesseract_cmd: str = "tesseract"
    tesseract_search_paths: tuple = ("/opt/homebrew/bin/tesseract",   # Apple Silicon Homebrew
                                     "/usr/local/bin/tesseract",      # Intel Homebrew
                                     "/usr/bin/tesseract")            # Linux
    language: str = "eng"
    # Page segmentation modes, tried in order until one finds text:
    #   3  = fully automatic layout (finds a clean single sign/label best)
    #   11 = sparse text, no layout (finds text the layout step misses, e.g. a
    #        light-on-dark sign inside a busy picture)
    # Found by testing: each mode missed a case the other one read correctly.
    # The second pass only runs when the first found nothing (costs ~0.15 s more).
    psm_modes: tuple = (3, 11)
    # Tesseract can use several CPU cores (OpenMP). One core keeps YOLO fast.
    cpu_threads: int = 1
    timeout_s: float = 8.0           # a read that takes longer is stopped

    # --- Image preprocessing ---------------------------------------------
    min_width: int = 1280            # smaller images are enlarged (Tesseract likes ~30 px tall letters)
    max_width: int = 2000            # bigger images are shrunk (speed)
    blur_threshold: float = 40.0     # variance of the Laplacian below this = too blurry to read
    binarize: str = "none"           # "none" | "otsu" | "adaptive" (see image_preprocessing.py)

    # --- Choosing a frame ---------------------------------------------------
    steady_wait_s: float = 2.5       # how long to wait for a steady, sharp frame before giving up

    # --- Text filtering -----------------------------------------------------
    min_word_conf: float = 60.0      # Tesseract confidence 0-100 per word
    min_text_chars: int = 3          # fewer letters/digits than this in total = no text
    max_spoken_words: int = 40       # long text is cut, so one read never talks for a minute
    duplicate_similarity: float = 0.85   # 0..1, difflib ratio; above = "the same text again"
    repeat_cooldown_s: float = 30.0      # continuous mode: same text is not repeated within this

    # --- Continuous reading mode ("keep reading") ---------------------------
    continuous_interval_s: float = 2.0   # time between reads
    answer_max_wait_s: float = 20.0      # an answer waits for SENSE to be quiet at most this long

    # --- Voice phrases (lower case, substring match, checked in this order) ---
    stop_phrases: tuple = ("stop reading", "reading off", "stop the reading", "end reading")
    continuous_phrases: tuple = ("keep reading", "start reading", "reading mode", "read continuously",
                                 "read everything you see")
    once_phrases: tuple = ("read this", "read that", "read it", "read the text", "read text",
                           "read the sign", "read the label", "read the paper", "read the page",
                           "what does this say", "what does that say", "what does it say",
                           "what's written", "what is written", "any text")
    extra: dict = field(default_factory=dict)   # room for experiments without changing the class
