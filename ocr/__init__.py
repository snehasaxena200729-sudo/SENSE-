"""
ocr - optional text reading for SENSE ("read this", "keep reading", "stop reading").

    config.py               all settings (OCRConfig)
    image_preprocessing.py  OpenCV clean-up of a frame before OCR
    ocr_engine.py           runs Tesseract as a separate process (non-blocking)
    text_filter.py          confidence filter, junk removal, repeat detection
    text_detector.py        WHERE is the text: EAST (OpenCV dnn) or a classic fallback
    text_regions.py         word boxes -> lines -> blocks (one label, one sign)
    target_selector.py      which block the user means ("read this")
    rectify.py              crop, deskew, perspective correction
    quality.py              clear / unclear / partial / unreadable, word voting
    camera_guidance.py      one useful tip when reading fails
    hud.py                  text boxes on the camera window after a read
    ocr_controller.py       the workflow SENSE talks to (OCRController)
    tests/                  independent tests: python -m unittest discover -s ocr/tests -t .

SENSE only needs OCRController. The package never opens a camera and never
starts a Python thread; it reads frames that sense_master.py already captured.
"""

from .config import OCRConfig
from .ocr_controller import OCRController, OCRMessage

__all__ = ["OCRController", "OCRConfig", "OCRMessage"]
