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
    # A reading is called "unclear" (and SENSE says so) when a spoken word scores
    # below sure_word_conf, or a real-looking word was dropped for low confidence
    # (but scored at least unclear_floor_conf; lower ones are mostly texture noise).
    # In testing, misread doses ("more than 6" for 8) scored 60-80, clean words 85+.
    sure_word_conf: float = 85.0
    unclear_floor_conf: float = 30.0
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
    # --- Text detection: WHERE is the text (ocr/text_detector.py) -------------
    # "read this" no longer sends the whole frame to Tesseract: text regions are
    # found first, one is chosen, and only that crop is read. False = the old
    # whole-frame behaviour (a safe fallback switch).
    use_text_detection: bool = True
    use_east: bool = True            # False = classic detector only (no model file)
    east_model_file: str = "frozen_east_text_detection.pb"
    east_url: str = ("https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/"
                     "master/frozen_east_text_detection.pb")
    east_sha256: str = "9b486f3c3eee77b4c8cc91a83892c37026cca7d29b79bf3b93772ccd2db58454"
    east_input_size: tuple = (960, 544)   # multiples of 32 (see docs/OCR.md for the speed trade-off)
    east_conf: float = 0.5
    east_nms: float = 0.4

    # --- Grouping words into blocks (ocr/text_regions.py) ----------------------
    line_gap_h: float = 1.6          # words on one line join if the gap < this x letter height
    block_gap_h: float = 1.2         # lines join into one block if the gap < this x letter height
    max_height_ratio: float = 2.2    # lines with very different letter sizes stay separate

    # --- Choosing the target (ocr/target_selector.py) --------------------------
    # Score = weighted mix of: near the frame centre, big letters (close to the
    # camera), sharp, and on an object YOLO sees. The winner must beat the
    # runner-up clearly, else SENSE says how to frame it (it never asks
    # "left or right" and never guesses).
    w_centre: float = 0.35
    w_size: float = 0.30
    w_sharp: float = 0.15
    w_object: float = 0.20
    cluster_gap_h: float = 3.0       # blocks closer than this (letter heights) are one sign/page
    ambiguous_ratio: float = 0.85    # runner-up >= this x best -> too close to call
    min_block_score: float = 0.12    # weaker blocks are ignored as background noise

    # --- Cropping, straightening and recipes (rectify.py, image_preprocessing.py)
    crop_margin_h: float = 0.6       # margin around the block, in letter heights
    deskew_min_deg: float = 2.0      # straighten only when tilted more than this
    use_perspective: bool = True     # warp a clear 4-corner label outline to a rectangle
    target_letter_px: int = 34       # crops are scaled so letters are about this tall
    max_crop_side: int = 2400
    max_attempts: int = 3            # recipes tried per read (each is one Tesseract run)
    good_score: float = 0.80         # a reading this good stops further attempts

    # --- Result quality (ocr/quality.py) ---------------------------------------
    min_coverage: float = 0.5        # fewer detected words read than this -> "only part of it"

    # --- Camera guidance (ocr/camera_guidance.py) ------------------------------
    min_letter_px: float = 11.0      # smaller letters: "move closer"
    max_letter_px: float = 160.0     # bigger: "move it a little further away"
    dark_mean: float = 60.0          # mean brightness below this: "more light"
    glare_fraction: float = 0.06     # share of blown-out pixels in the block: "glare"
    tilt_deg: float = 15.0           # text slope above this: "turn it to face the camera"
    edge_margin: float = 0.02        # block this close to the frame edge = text is cut off

    # --- HUD ------------------------------------------------------------------
    show_boxes_s: float = 6.0        # text boxes stay on screen this long after a read

    extra: dict = field(default_factory=dict)   # room for experiments without changing the class
