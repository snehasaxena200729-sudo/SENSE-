"""Home-map settings and word lists in one place, with the reason for each."""

from dataclasses import dataclass, field

# Room words people say, including Indian English ones. They only decide
# whether "take me to the X" is about a ROOM; the user's own name is kept.
ROOM_WORDS = (
    "kitchen", "bedroom", "bathroom", "washroom", "restroom", "toilet", "living room",
    "hall", "hallway", "corridor", "passage", "lounge", "drawing room", "dining room",
    "study", "study room", "office", "balcony", "terrace", "veranda", "verandah",
    "garage", "guest room", "master bedroom", "kids room", "pooja room", "puja room",
    "prayer room", "store room", "storeroom", "laundry", "laundry room", "utility room",
    "lobby", "entrance", "porch", "attic", "basement", "playroom", "nursery", "library",
    "gym", "classroom", "lab",
)

# Whisper splits, joins or misspells compound words.
SPOKEN_FIXES = {
    "bed room": "bedroom", "bath room": "bathroom", "wash room": "washroom",
    "rest room": "restroom", "store room": "storeroom", "hall way": "hallway",
    "livingroom": "living room", "diningroom": "dining room", "guestroom": "guest room",
    "drawingroom": "drawing room", "poojaroom": "pooja room", "pujaroom": "puja room",
    "kitchin": "kitchen", "kichen": "kitchen", "bedrooms": "bedroom", "kitchens": "kitchen",
}

# Never room names: SENSE's object words and a few fillers. "Take me to the
# door" or "take me to my bottle" must stay with the existing find command.
NOT_ROOMS = {
    "phone", "remote", "cup", "bottle", "book", "laptop", "mouse", "bag", "chair", "couch",
    "bed", "table", "plant", "tv", "dog", "cat", "suitcase", "bench", "bicycle", "door",
    "stairs", "door handle", "keys", "wallet", "glasses", "person", "it", "this", "that",
    "here", "there", "me", "him", "her", "them",
}


@dataclass
class MappingConfig:
    # "Where am I?" repeats the room you named only if you named it in this
    # run of SENSE and within this many seconds. Phase 0 cannot recognise
    # rooms, and an old answer could send a blind user the wrong way.
    declared_room_valid_s: float = 30 * 60

    # Longest room name accepted, in words ("master bedroom", "sneha's room").
    max_name_words: int = 3

    # Misheard room words: difflib similarity needed (0..1). 0.8 accepts
    # "bedrom" -> "bedroom" but not "bathroom" -> "bedroom" (0.67).
    fuzzy_cutoff: float = 0.8

    room_words: tuple = ROOM_WORDS
    spoken_fixes: dict = field(default_factory=lambda: dict(SPOKEN_FIXES))
    not_rooms: set = field(default_factory=lambda: set(NOT_ROOMS))
