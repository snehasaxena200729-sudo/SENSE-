"""
ocr/hud.py - draw the text boxes on SENSE's camera window.

Only after the user asks to read (R or "read this"), for config.show_boxes_s:
  thin yellow   every text block the detector found
  thick green   the block SENSE chose and is reading
  red label     a block SENSE could not read
Drawing is a few cv2.polylines calls (well under 1 ms).
"""

import cv2
import numpy as np

YELLOW, GREEN, RED = (0, 220, 255), (0, 200, 0), (0, 0, 230)


def draw(frame, blocks, chosen=None, label="TEXT", failed=False):
    for b in blocks:
        cv2.polylines(frame, [np.int32(b.quad)], True, YELLOW, 1)
    if chosen is not None:
        color = RED if failed else GREEN
        cv2.polylines(frame, [np.int32(chosen.quad)], True, color, 3)
        x, y = int(chosen.quad[:, 0].min()), int(chosen.quad[:, 1].min())
        cv2.putText(frame, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
