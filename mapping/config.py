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

    # --- Phase 1: teaching a room by turning round once -----------------
    view_step_deg: float = 20.0          # one saved view per 20 degrees -> 18 per full turn
    capture_min_interval_s: float = 0.25  # at most 4 views a second (main-thread time)
    min_sharpness: float = 12.0          # Laplacian variance on a 160 px frame; below = motion blur
    done_bins: int = 16                  # 16 of 18 directions seen = done (a few may be blurred)
    min_views: int = 6                   # fewer views than this after a timeout = try again
    min_stop_views: int = 3              # "stop" early keeps what was seen if at least this many
    max_teach_s: float = 45.0
    nudge_after_s: float = 6.0           # still not turning after this: "turn slowly to your right"

    # --- Phase 1: recognising the room ----------------------------------
    recognise_every_n: int = 10          # check on every 10th frame, only with the camera steady
    knn_two: bool = False                # room score = best view (True: mean of the best two)
    # likelihood = exp(beta * (similarity - tau)); tau is where a room and "a place
    # I don't know" are equally likely. Measured on real indoor photos (docs/HOME_MAP.md).
    beta_centred: float = 12.0
    unknown_sim_centred: float = 0.32
    beta_raw: float = 20.0               # only one room taught: no centering possible
    unknown_sim_raw: float = 0.62
    sure_sim_centred: float = 0.40       # "You are in ..." needs this image match, or corners
    sure_sim_raw: float = 0.80
    orb_min_inliers: int = 15            # this many geometry-consistent corner matches = same spot
    orb_max_boost: float = 4.0
    landmark_clip: tuple = (0.33, 3.0)   # landmarks can shift the odds, never decide alone
    likelihood_clip: tuple = (0.25, 5.0)  # one odd frame can't flip the room; 2-3 good ones can
    stay_prob: float = 0.9               # between two checks you most likely stay in the room
    sure_prob: float = 0.75              # "You are in the kitchen."
    likely_prob: float = 0.5             # "I think you are in the kitchen."
    belief_fresh_s: float = 4.0          # older than this when asked: check again first
    landmark_min_conf: float = 0.4
    landmark_classes: tuple = ("bed", "couch", "tv", "dining table", "refrigerator", "oven",
                               "microwave", "sink", "toilet", "chair", "potted plant", "bench",
                               "toaster", "clock", "vase", "laptop", "book")

    room_words: tuple = ROOM_WORDS
    spoken_fixes: dict = field(default_factory=lambda: dict(SPOKEN_FIXES))
    not_rooms: set = field(default_factory=lambda: set(NOT_ROOMS))
