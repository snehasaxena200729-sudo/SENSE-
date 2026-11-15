"""
spatial.py - World-anchored directions and spatial relations (Roadmap E4, O3)

WHY
---
The camera only knows where things are *relative to where it is pointing now*.
SENSE estimates how far the user has turned (the "heading") with optical flow.
Old code stored only "left / front / right" and assumed the user was facing 0
degrees when the object was seen - wrong if they had already turned.

THE FIX: store a WORLD angle for every observation.

    world_angle = angle_inside_frame + heading_at_that_moment

and at question time compute

    relative = world_angle - heading_now      (wrapped to -180..180)

Positive angles mean "to the right". This is plain dead reckoning; it drifts a
little over time because optical-flow errors add up, and it resets each run
(so we tag every observation with a session id and only give turn advice for
the current session).
"""

CAMERA_FOV_DEG = 65.0  # same assumption as the heading code in sense_master.py


def wrap_angle(deg):
    """Wrap any angle into the range [-180, 180). Always returns a plain float
    (numpy floats would be stored by SQLite as unreadable bytes)."""
    return float(((float(deg) + 180.0) % 360.0) - 180.0)


def pixel_to_angle(x, frame_width, fov=CAMERA_FOV_DEG):
    """Horizontal angle of pixel column x relative to the camera's centre line."""
    return (x / float(frame_width) - 0.5) * fov


def world_angle(x, frame_width, heading, fov=CAMERA_FOV_DEG):
    return wrap_angle(pixel_to_angle(x, frame_width, fov) + heading)


def relative_angle(world_deg, heading_now):
    return wrap_angle(world_deg - heading_now)


def direction_phrase(rel_deg):
    """Turn a relative angle into words a listener can act on."""
    a = abs(rel_deg)
    if a < 11:           # roughly the centre third of the frame
        return "in front of you"
    if a > 135:
        return "behind you"
    return "on your right" if rel_deg > 0 else "on your left"


def turn_instruction(rel_deg):
    """'turn about 90 degrees to your right' (None if it's already ahead)."""
    a = abs(rel_deg)
    if a < 25:
        return None
    rounded = int(round(a / 15.0) * 15)
    if rounded >= 165:
        return "turn around"
    side = "right" if rel_deg > 0 else "left"
    return f"turn about {rounded} degrees to your {side}"


def is_in_view(world_deg, heading_now, fov=CAMERA_FOV_DEG, margin=5.0):
    """Would an object at this world angle be inside the current camera view?"""
    return abs(relative_angle(world_deg, heading_now)) < (fov / 2.0 - margin)


def nearest_neighbour(target, others, max_angle_gap=12.0, max_time_gap=30.0):
    """Find the closest other object to `target` (Roadmap O3).

    target / others: dicts with "name", "world_angle", "last_y", "last_seen".
    Closeness = angular gap (dominant) + vertical pixel gap (secondary).
    Only objects seen around the same time are considered, because a laptop
    seen yesterday says nothing about where the phone is today.
    """
    if target.get("world_angle") is None:
        return None
    best, best_cost = None, None
    for o in others:
        if o["name"] == target["name"] or o.get("world_angle") is None:
            continue
        if abs((o.get("last_seen") or 0) - (target.get("last_seen") or 0)) > max_time_gap:
            continue
        gap = abs(wrap_angle(o["world_angle"] - target["world_angle"]))
        if gap > max_angle_gap:
            continue
        cost = gap + abs((o.get("last_y") or 0) - (target.get("last_y") or 0)) / 40.0
        if best_cost is None or cost < best_cost:
            best, best_cost = o, cost
    return best


def relation_phrase(target, neighbour):
    """'next to the laptop' / 'to the left of the laptop'."""
    if neighbour is None:
        return None
    gap = wrap_angle(target["world_angle"] - neighbour["world_angle"])
    name = neighbour["name"]
    if abs(gap) < 4:
        return f"next to the {name}"
    return f"to the {'right' if gap > 0 else 'left'} of the {name}"
