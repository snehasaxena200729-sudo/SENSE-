"""
ocr/image_preprocessing.py - make a camera frame easier for Tesseract to read.

Pure OpenCV + NumPy functions: no camera, no Tesseract, no state. Each step is
a classic technique:

  1. GRAYSCALE      colour does not help OCR; one channel is 3x less data.
  2. SHARPNESS      variance of the Laplacian (a second-derivative edge filter).
                    Sharp edges -> large values, motion blur -> small values.
                    Reading a blurry frame gives confident-looking garbage, so a
                    blurry frame is skipped and SENSE asks the user to hold still.
  3. RESIZE         Tesseract's LSTM works best around 20-40 px letter height.
                    Webcam frames of a label held at arm's length have small
                    letters, so small frames are enlarged (bicubic interpolation).
  4. CLAHE          Contrast Limited Adaptive Histogram Equalisation: boosts
                    contrast tile by tile, so a dim corner and a bright corner
                    both become readable, without blowing up noise (the "limit").
  5. POLARITY       Tesseract prefers dark text on a light background. If most
                    of the image is dark (white text on a dark sign), invert it.
  6. BINARIZE       optional: Otsu (one global threshold chosen from the
                    histogram) or adaptive (a threshold per neighbourhood, copes
                    with shadows). Off by default because Tesseract 4/5 already
                    binarizes internally, and an early hard threshold can erase
                    thin letters. Try "adaptive" for strongly uneven light.
"""

import cv2
import numpy as np


def to_gray(frame):
    """BGR (or already gray) image -> single-channel uint8."""
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def sharpness(gray):
    """Variance of the Laplacian: higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_blurry(gray, threshold):
    return sharpness(gray) < threshold


def resize_for_ocr(gray, min_width, max_width):
    h, w = gray.shape[:2]
    if w < min_width:
        scale = min_width / float(w)
        return cv2.resize(gray, (min_width, int(round(h * scale))), interpolation=cv2.INTER_CUBIC)
    if w > max_width:
        scale = max_width / float(w)
        return cv2.resize(gray, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return gray


def enhance_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def ensure_dark_text(gray):
    """Invert when the image is mostly dark (light text on a dark background)."""
    return cv2.bitwise_not(gray) if float(np.mean(gray)) < 110 else gray


def binarize(gray, method):
    if method == "otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out
    if method == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    return gray


def prepare(frame, config):
    """Full pipeline. Returns (image_for_tesseract, sharpness_score).

    The sharpness is measured on the ORIGINAL-size gray image, before resizing,
    because enlarging an image smooths it and would make every frame look blurry.
    The input frame is never modified.
    """
    gray = to_gray(frame)
    score = sharpness(gray)
    img = resize_for_ocr(gray, config.min_width, config.max_width)
    img = enhance_contrast(img)
    img = ensure_dark_text(img)
    img = binarize(img, config.binarize)
    return img, score


# ---------------------------------------------------------------------------
# Crop preprocessing (text detection path). A crop holds ONE text block, so the
# steps that break whole frames (hard thresholds on background texture) can be
# tried here. Instead of one fixed pipeline there are small "recipes"; the
# controller tries up to config.max_attempts of them and keeps the best reading.
#
#   clahe     today's recipe: local contrast boost (works on most clean prints)
#   sharpen   unsharp mask: image + 0.8 * (image - blurred image). Restores
#             edges softened by slight defocus or motion. Amplifies noise too,
#             so it is not first on grainy images.
#   otsu      one global black/white threshold picked from the histogram.
#             Good for high-contrast labels; removes faint background print.
#   adaptive  a threshold per neighbourhood (block size ~ 1.5 letter heights),
#             for labels with a shadow across them.
#   denoise   non-local means: replaces each pixel by an average of pixels
#             whose SURROUNDING PATCH looks similar, so letter edges survive
#             while sensor grain (dim rooms) is smoothed. ~20-60 ms on a crop.
#
# All recipes end with "dark text on light" polarity and a white border:
# Tesseract's line finder works better when text does not touch the edge.
# ---------------------------------------------------------------------------

def scale_to_letter_height(gray, letter_h, target_px, max_side):
    """Resize so letters are about target_px tall (Tesseract's sweet spot)."""
    if letter_h <= 0:
        return gray, 1.0
    s = target_px / float(letter_h)
    h, w = gray.shape[:2]
    s = min(s, max_side / float(max(h, w)))
    s = max(s, 0.25)
    if abs(s - 1.0) < 0.05:
        return gray, 1.0
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    return cv2.resize(gray, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=interp), s


def unsharp(gray, amount=0.8, sigma=1.5):
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)


def denoise(gray):
    return cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)


def adaptive(gray, letter_px):
    block = int(letter_px * 1.5) | 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                 max(block, 15), 10)


def image_stats(gray):
    """Brightness, grain and shadow numbers used to order the recipes."""
    mean = float(np.mean(gray))
    noise = float(np.std(gray.astype(np.int16) - cv2.medianBlur(gray, 3).astype(np.int16)))
    k = max(3, (min(gray.shape[:2]) // 4) | 1)
    illum = cv2.GaussianBlur(gray, (k, k), 0).astype(np.float32)
    uneven = float(np.std(illum) / (np.mean(illum) + 1e-6))
    return {"mean": mean, "noise": noise, "uneven": uneven}


def plan_recipes(stats):
    """Recipe names in the order to try them (decided from tools/ocr_eval.py runs)."""
    if stats["mean"] < 70 or stats["noise"] > 9:
        return ["denoise", "clahe", "otsu"]
    if stats["uneven"] > 0.25:
        return ["clahe", "adaptive", "sharpen"]
    return ["clahe", "sharpen", "otsu"]


def apply_recipe(name, gray, letter_px):
    if name == "denoise":
        img = enhance_contrast(denoise(gray))
    elif name == "sharpen":
        img = unsharp(enhance_contrast(gray))
    elif name == "otsu":
        img = binarize(enhance_contrast(gray), "otsu")
    elif name == "adaptive":
        img = adaptive(enhance_contrast(gray), letter_px)
    else:
        img = enhance_contrast(gray)
    img = ensure_dark_text(img)
    pad = max(10, int(letter_px))
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255), pad
