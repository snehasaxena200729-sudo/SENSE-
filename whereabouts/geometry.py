"""Bounding-box helpers. A box is (x1, y1, x2, y2) in pixels, y grows downwards."""


def width(b):
    return max(0.0, b[2] - b[0])


def height(b):
    return max(0.0, b[3] - b[1])


def area(b):
    return width(b) * height(b)


def centre_x(b):
    return (b[0] + b[2]) / 2.0


def bottom_centre(b):
    """Where an item touches whatever it rests on (in the picture)."""
    return centre_x(b), b[3]


def intersection(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def horizontal_overlap(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def horizontal_gap(a, b):
    """Empty space between the boxes' sides (0 when they overlap sideways)."""
    return max(0.0, b[0] - a[2], a[0] - b[2])
