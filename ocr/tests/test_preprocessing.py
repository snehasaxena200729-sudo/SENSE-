import unittest

import numpy as np

from ocr.config import OCRConfig
from ocr.image_preprocessing import (binarize, ensure_dark_text, prepare, resize_for_ocr,
                                     sharpness, to_gray)
from ocr.tests.helpers import scene


class PreprocessingTests(unittest.TestCase):
    def test_gray_and_input_not_modified(self):
        frame = scene(["EXIT"])
        before = frame.copy()
        img, _ = prepare(frame, OCRConfig())
        self.assertEqual(img.ndim, 2)
        self.assertTrue(np.array_equal(frame, before), "prepare() must not draw on SENSE's frame")

    def test_blur_detection_separates_sharp_and_blurry(self):
        sharp = sharpness(to_gray(scene(["MEDICINE"])))
        blurry = sharpness(to_gray(scene(["MEDICINE"], blur=15)))
        self.assertGreater(sharp, OCRConfig().blur_threshold)
        self.assertLess(blurry, OCRConfig().blur_threshold)

    def test_small_frames_are_enlarged_big_ones_shrunk(self):
        cfg = OCRConfig()
        small = resize_for_ocr(np.zeros((480, 640), np.uint8), cfg.min_width, cfg.max_width)
        big = resize_for_ocr(np.zeros((2000, 4000), np.uint8), cfg.min_width, cfg.max_width)
        self.assertEqual(small.shape, (960, 1280))
        self.assertEqual(big.shape[1], cfg.max_width)

    def test_light_text_on_dark_is_inverted(self):
        dark = np.full((100, 100), 20, np.uint8)
        self.assertGreater(ensure_dark_text(dark).mean(), 200)
        light = np.full((100, 100), 220, np.uint8)
        self.assertEqual(ensure_dark_text(light).mean(), 220)

    def test_binarize_modes(self):
        g = to_gray(scene(["EXIT"]))
        for mode in ("otsu", "adaptive"):
            self.assertTrue(set(np.unique(binarize(g, mode))) <= {0, 255})
        self.assertIs(binarize(g, "none"), g)


if __name__ == "__main__":
    unittest.main()
