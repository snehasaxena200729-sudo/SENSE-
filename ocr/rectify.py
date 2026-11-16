"""
ocr/rectify.py - cut the chosen text block out of the frame and straighten it.

Tesseract assumes level lines of text. Two common camera problems break that:

  TILT (in-plane rotation)  the label is rotated, like a picture hanging
           crooked. EAST measures each word's angle, so we rotate the crop by
           the block's median angle ("deskewing"). Implemented as one
           perspective warp from the block's rotated rectangle to an upright
           rectangle, which crops and rotates in a single resampling step.

  PERSPECTIVE  the label leans away from the camera, so a rectangle looks like
           a trapezoid (far side shorter). If we can find the label's own
           4-corner outline (edges -> contours -> polygon approximation with
           4 corners), a homography (3x3 matrix mapping any 4 points to any 4
           points) warps it back to a rectangle. Only used when such an outline
           is found clearly and is really non-rectangular; otherwise we keep
           the rotated crop, because a wrong outline would distort good text.

Every function returns the 3x3 matrix M that maps FRAME coordinates to CROP
coordinates, so quality.py can find the detected words inside the crop.
"""

import cv2
import numpy as np


def order_corners(pts):
    """4 points -> top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()        # y - x
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    np.float32)


def _expand(quad, margin):
    """Push each corner `margin` px away from the centre along both box axes."""
    q = order_corners(quad)
    ux = q[1] - q[0]
    uy = q[3] - q[0]
    ux /= (np.linalg.norm(ux) + 1e-6)
    uy /= (np.linalg.norm(uy) + 1e-6)
    signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    return np.array([q[i] + sx * margin * ux + sy * margin * uy for i, (sx, sy) in enumerate(signs)],
                    np.float32)


def _warp(gray, src):
    tl, tr, br, bl = src
    w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    w, h = max(w, 8), max(h, 8)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), M


def find_label_outline(gray, block):
    """A clear 4-corner outline around the block (the label's edge), or None."""
    fh, fw = gray.shape[:2]
    x, y, w, h = block.rect
    pad = int(max(w, h) * 0.6)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(fw, x + w + pad), min(fh, y + h + pad)
    win = gray[y0:y1, x0:x1]
    if win.size == 0:
        return None
    edges = cv2.Canny(cv2.GaussianBlur(win, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    block_area = float(w * h)
    cx, cy = x + w / 2.0 - x0, y + h / 2.0 - y0
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1.1 * block_area or area > 6 * block_area:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        if cv2.pointPolygonTest(approx, (float(cx), float(cy)), False) < 0:
            continue
        if best is None or area < best[0]:       # tightest outline around the text
            best = (area, approx.reshape(4, 2).astype(np.float32) + [x0, y0])
    return order_corners(best[1]) if best else None


def _is_trapezoid(q, tol=0.08):
    tl, tr, br, bl = q
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    return (abs(top - bottom) / max(top, bottom) > tol) or (abs(left - right) / max(left, right) > tol)


def crop_block(frame_gray, block, config):
    """-> (crop, M, how) with how in {'crop', 'deskew', 'perspective'}."""
    margin = config.crop_margin_h * block.letter_h
    if config.use_perspective:
        outline = find_label_outline(frame_gray, block)
        if outline is not None and _is_trapezoid(outline):
            crop, M = _warp(frame_gray, outline)
            return crop, M, "perspective"
    if abs(block.angle) >= config.deskew_min_deg:
        crop, M = _warp(frame_gray, _expand(block.quad, margin))
        return crop, M, "deskew"
    fh, fw = frame_gray.shape[:2]
    x, y, w, h = block.rect
    m = int(round(margin))
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(fw, x + w + m), min(fh, y + h + m)
    M = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    return frame_gray[y0:y1, x0:x1].copy(), M, "crop"


def map_points(M, pts):
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(M, np.float64)).reshape(-1, 2)
