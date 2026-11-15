"""
face_id.py - Who is this person? Verified face identity (audit fix 4).

HOW IT WORKS (two small neural networks that ship with OpenCV's API)
--------------------------------------------------------------------
1. YuNet (cv2.FaceDetectorYN, 0.2 MB): finds faces and 5 landmarks
   (two eyes, nose tip, two mouth corners).
2. SFace (cv2.FaceRecognizerSF, 37 MB): aligns the face using the landmarks
   and turns it into an "embedding", 128 numbers. Photos of the SAME person
   give embeddings pointing in nearly the same direction.

   similarity = cosine(a, b) = a . b / (|a| |b|)      (1 = identical)

   OpenCV's published threshold for SFace is 0.363: above it, same person.
   (Measured on the LFW benchmark; on a webcam in a dim room it is less sure,
   which is why SENSE also demands a margin and repeated agreement.)

WHAT SENSE DEMANDS BEFORE IT SAYS A NAME
----------------------------------------
* face quality: detector score >= 0.9, eyes >= 28 px apart, roughly frontal
* similarity >= 0.363 to that person AND at least MARGIN above anyone else
* people.PersonIdentityGate: 3 of the last 5 checks on that track agree
Anything less -> "a person". Being silent about a name is correct behaviour.

No new pip package: both networks run in OpenCV's DNN module, which is
already installed (opencv-python >= 4.5.4). The two .onnx files download once
into models/, with a checksum check; after that it is fully offline.
"""

import hashlib
import os
import urllib.request

import numpy as np

try:
    import cv2
except ImportError:          # tests can run without OpenCV
    cv2 = None

from people import NO_FACE, UNKNOWN

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MODELS = {
    "detector": ("face_detection_yunet_2023mar.onnx",
                 _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
                 "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "recognizer": ("face_recognition_sface_2021dec.onnx",
                   _ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
                   "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}

MATCH_THRESHOLD = 0.363   # OpenCV's cosine threshold for SFace
MARGIN = 0.05             # best person must beat the next best by this much
MIN_FACE_SCORE = 0.90
MIN_EYE_PX = 28           # smaller faces carry too little detail to identify
MAX_YAW_RATIO = 0.25      # nose offset from the eye midpoint / eye distance


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_paths():
    return {k: os.path.join(MODEL_DIR, v[0]) for k, v in MODELS.items()}


def ensure_models(download=True):
    """Make sure both model files exist and are intact. Returns True if ready."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for key, (fname, url, sha) in MODELS.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path) and _sha256(path) == sha:
            continue
        if not download:
            return False
        print(f"🙂 Downloading {fname} (one time only)...")
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        if _sha256(tmp) != sha:
            os.remove(tmp)
            raise ValueError(f"{fname}: checksum mismatch, download rejected")
        os.replace(tmp, path)
    return True


def cosine(a, b):
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def face_quality_ok(face):
    """face: YuNet row [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y, ..., score]."""
    if face[14] < MIN_FACE_SCORE:
        return False
    eye_dx, eye_dy = face[6] - face[4], face[7] - face[5]
    eye = float(np.hypot(eye_dx, eye_dy))
    if eye < MIN_EYE_PX:
        return False
    mid_x = (face[4] + face[6]) / 2.0
    return abs(face[8] - mid_x) / eye <= MAX_YAW_RATIO


def eye_distance(face):
    return float(np.hypot(face[6] - face[4], face[7] - face[5]))


def decide(embedding, profiles):
    """Compare one embedding with every enrolled person.

    profiles: {name: array (k, 128)}. Returns (name or UNKNOWN, best score, runner-up score).
    """
    scores = sorted(((max(cosine(embedding, e) for e in embs), name)
                     for name, embs in profiles.items()), reverse=True)
    if not scores:
        return UNKNOWN, 0.0, 0.0
    best, name = scores[0]
    second = scores[1][0] if len(scores) > 1 else 0.0
    if best >= MATCH_THRESHOLD and best - second >= MARGIN:
        return name, best, second
    return UNKNOWN, best, second


class FaceID:
    """Face detection + recognition on person crops. Main thread only."""

    def __init__(self, paths):
        self.detector = cv2.FaceDetectorYN.create(paths["detector"], "", (320, 320), 0.8, 0.3, 5)
        self.recognizer = cv2.FaceRecognizerSF.create(paths["recognizer"], "")
        self.profiles = {}      # name -> array (k, 128)
        self.since = {}         # name -> time of first enrolment

    @classmethod
    def create(cls, download=True):
        """Returns a FaceID, or None (and says why) if it cannot run."""
        if cv2 is None or not hasattr(cv2, "FaceDetectorYN"):
            print("🙂 Face recognition off: OpenCV is too old (needs 4.5.4+).")
            return None
        try:
            if not ensure_models(download):
                return None
            return cls(model_paths())
        except Exception as e:  # noqa: BLE001  (no internet on first run, etc.)
            print(f"🙂 Face recognition off ({e}). Connect to the internet once to download the face models.")
            return None

    @classmethod
    def load_if_available(cls):
        return cls.create(download=False)

    def set_profiles(self, loaded):
        """loaded: memory_db.load_face_profiles() output."""
        self.profiles = {n: d["embeddings"] for n, d in loaded.items()}
        self.since = {n: d["since"] for n, d in loaded.items()}

    # --- per-frame work ----------------------------------------------------
    def best_face(self, frame, box):
        """Largest face inside the top part of a person box (or None)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        y2 = min(y2, y1 + max(int((x2 - x1) * 1.6), 40))    # head is in the upper part
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 24 or y2 - y1 < 24:
            return None, None
        crop = frame[y1:y2, x1:x2]
        self.detector.setInputSize((crop.shape[1], crop.shape[0]))
        _, faces = self.detector.detect(crop)
        if faces is None or len(faces) == 0:
            return None, None
        face = max(faces, key=lambda f: f[2] * f[3])
        return crop, face

    def eye_distance_px(self, frame, box):
        crop, face = self.best_face(frame, box)
        if face is None or face[14] < MIN_FACE_SCORE:
            return None
        return eye_distance(face)

    def embed(self, crop, face):
        aligned = self.recognizer.alignCrop(crop, face)
        return self.recognizer.feature(aligned).ravel().astype(np.float32)

    def check(self, frame, box):
        """One identity check of a person box.

        Returns (result, eye_px, embedding): result is NO_FACE (nothing usable),
        UNKNOWN (clear face, nobody enrolled matches) or a verified name.
        """
        crop, face = self.best_face(frame, box)
        if face is None:
            return NO_FACE, None, None
        eye_px = eye_distance(face) if face[14] >= MIN_FACE_SCORE else None
        if not face_quality_ok(face):
            return NO_FACE, eye_px, None
        emb = self.embed(crop, face)
        if not self.profiles:
            return UNKNOWN, eye_px, emb
        name, best, second = decide(emb, self.profiles)
        return name, eye_px, emb


class Enrolment:
    """'This is Sneha': collect several good face samples over a few seconds,
    without blocking the main loop (the voice path must stay fast)."""
    SAMPLES = 5
    MIN_SAMPLES = 3
    TIMEOUT_S = 4.0

    def __init__(self, name, track_id, now):
        self.name, self.track_id, self.started = name, track_id, now
        self.samples = []

    def add(self, embedding):
        if embedding is not None and len(self.samples) < self.SAMPLES:
            self.samples.append(embedding)

    def finished(self, now):
        return len(self.samples) >= self.SAMPLES or now - self.started > self.TIMEOUT_S

    def ok(self):
        return len(self.samples) >= self.MIN_SAMPLES

    def consistent(self):
        """All samples must look like one person (guards against the track
        jumping to someone else during enrolment)."""
        s = self.samples
        return all(cosine(s[i], s[j]) >= MATCH_THRESHOLD
                   for i in range(len(s)) for j in range(i + 1, len(s)))
