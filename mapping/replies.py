"""replies.py - every sentence the home map speaks, in one place."""


def spoken_room(name):
    """'kitchen' -> 'the kitchen', "sneha's room" -> "Sneha's room", 'my room' -> 'your room'."""
    n = name.strip().lower()
    if n.startswith("my "):
        return "your " + n[3:]
    if "'s " in n or n.endswith("'s"):
        return n[0].upper() + n[1:]
    return "the " + n


def join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def minutes_ago(seconds):
    m = int(round(seconds / 60.0))
    if m < 1:
        return "just now"
    return "1 minute ago" if m == 1 else f"{m} minutes ago"


HOW_TO_TEACH = "To add one, stand in the room and say: this room is the kitchen."


def no_rooms():
    return "No rooms are registered yet. " + HOW_TO_TEACH


def room_list(names):
    if not names:
        return no_rooms()
    rooms = [spoken_room(n) for n in names]
    count = "1 room" if len(rooms) == 1 else f"{len(rooms)} rooms"
    return f"I know {count}: {join(rooms)}."


def teach_missed(heard):
    return ("I didn't catch the room's name. Say: this room is the kitchen."
            if not heard else f"I heard {heard}, but that isn't a room name I can use. "
                              f"Say: this room is the kitchen.")


def taught(name, is_new, total):
    if is_new:
        more = "" if total == 1 else f" You now have {total} rooms."
        return (f"Okay, I saved {spoken_room(name)} as a room.{more} "
                f"I can't recognise rooms by sight yet, so tell me when you move.")
    return f"Okay, you are in {spoken_room(name)}."


def where_am_i(declared, age_s, any_rooms):
    if declared is not None:
        return (f"The last room you told me was {spoken_room(declared)}, {minutes_ago(age_s)}. "
                f"I can't recognise rooms by sight yet.")
    if not any_rooms:
        return "I don't know which room you are in. " + no_rooms()
    return ("I don't know which room you are in. I can't recognise rooms by sight yet. "
            "You can tell me: this room is the kitchen.")


def go(name, known, current):
    if current and current == name:
        return f"You told me you are already in {spoken_room(name)}."
    if not known:
        return (f"I don't know {spoken_room(name)} yet. When you are in it, say: "
                f"this room is {name if name.startswith('my ') else spoken_room(name)}.")
    return (f"I know {spoken_room(name)}, but I can't guide you between rooms yet. "
            f"That comes in a later update.")


# --- Phase 1 ------------------------------------------------------------------

def teach_start(name, relearn=False):
    lead = f"Okay, let's relearn {spoken_room(name)}." if relearn else f"Okay, {spoken_room(name)}."
    return lead + " Now turn slowly all the way round. I'll tell you when I'm done."


def teach_nudge():
    return "Please turn slowly to your right, all the way round."


def teach_halfway():
    return "Halfway there. Keep turning slowly."


def teach_done(name, n_views):
    return f"Done. I've learned {spoken_room(name)} from {n_views} views."


def teach_failed(name):
    return (f"I couldn't see enough of {spoken_room(name)}. Please try again: "
            f"say relearn this room, then turn slowly all the way round.")


def teach_cancelled(name):
    return f"Okay, I stopped learning {spoken_room(name)}. Nothing new was saved."


def still_teaching(name):
    return f"I'm still learning {spoken_room(name)}. Keep turning slowly, or say stop."


def already_taught(name):
    return f"Okay, you are in {spoken_room(name)}."


def recognised(name, sure):
    return f"You are in {spoken_room(name)}." if sure else f"I think you are in {spoken_room(name)}, but I'm not completely sure."


def unknown_place():
    return ("This doesn't look like any room you've taught me. To teach it, "
            "say: this room is, and then its name.")


def not_sure(declared=None, age_s=0):
    if declared:
        return (f"I can't tell from what I see right now. The last room you told me was "
                f"{spoken_room(declared)}, {minutes_ago(age_s)}.")
    return "I'm not sure which room this is. Hold the camera steady for a moment, then ask again."


def relearn_which():
    return "Which room is this? Say: this room is the kitchen."


def relearn_unknown(name):
    return f"I don't know {spoken_room(name)} yet. To teach it, say: this room is {name if name.startswith('my ') else spoken_room(name)}."


def no_camera_yet():
    return "I can't see anything yet. Please wait a moment, then try again."
