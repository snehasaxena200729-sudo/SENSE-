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
