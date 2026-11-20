"""
fingerprint.py - turn one camera frame into numbers that describe the room.

Two kinds of numbers, stored for each saved view (no photo is ever kept):

1. A global "embedding" from a small neural network, MobileNetV2 (ImageNet),
   run with OpenCV's DNN module, which SENSE already uses for faces. We take
   the 1280 numbers just before the classifier (after global average pooling).
   They describe WHAT the picture looks like (cupboards, tiles, a bed...), not
   where each pixel is, so a slightly different view of the same room gives a
   similar vector. Similarity = cosine of the angle between two vectors.
   The 14 MB model file downloads once into models/ (checksum checked), like
   the face models; after that everything is offline.

2. ORB corner features: up to 400 small corner patterns (32 bytes each) with
   their positions. Two views of the same spot share many corners in a
   consistent geometric layout; two similar-looking white walls do not. Used
   to CHECK the top candidates (see recognise.py).

Neither can be turned back into a photo of your home.
"""

import hashlib
import io
import os
import urllib.request

import numpy as np

try:
    import cv2
except ImportError:            # parts of the tests run without OpenCV
    cv2 = None

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_FILE = "image_classification_mobilenetv2_2022apr.onnx"
MODEL_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
             "image_classification_mobilenet/" + MODEL_FILE)
MODEL_SHA256 = "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"
POOL_TENSOR = "464"            # output of GlobalAveragePool_97 in that file
POOL_LAYER = "GlobalAveragePool_97"
EMBED_DIM = 1280
ORB_WIDTH = 320                # frames are shrunk to this width for ORB (speed)
ORB_FEATURES = 400


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(download=True, model_dir=MODEL_DIR):
    """Path of the model file, downloading it once if needed; None if unavailable."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, MODEL_FILE)
    if os.path.exists(path) and _sha256(path) == MODEL_SHA256:
        return path
    if not download:
        return None
    print(f"🏠 Downloading {MODEL_FILE} (14 MB, one time only)...")
    tmp = path + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    if _sha256(tmp) != MODEL_SHA256:
        os.remove(tmp)
        raise ValueError(f"{MODEL_FILE}: checksum mismatch, download rejected")
    os.replace(tmp, path)
    return path


# --- a tiny ONNX edit, so OpenCV 5 also returns the pooled layer --------------
# OpenCV 4 can read any layer by name. OpenCV 5's new engine only returns the
# graph's declared outputs, so we declare one more output (tensor "464") in the
# model bytes we load (the file on disk is not changed). ONNX files are
# protobuf: ModelProto field 7 is the graph; GraphProto field 12 is "output";
# ValueInfoProto field 1 is the name.

def _varint(n):
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _read_varint(buf, i):
    shift = n = 0
    while True:
        b = buf[i]
        i += 1
        n |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return n, i


def add_graph_output(model_bytes, tensor_name):
    out, i = bytearray(), 0
    while i < len(model_bytes):
        key, j = _read_varint(model_bytes, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            _, k = _read_varint(model_bytes, j)
            out += model_bytes[i:k]
            i = k
        elif wire == 2:
            ln, k = _read_varint(model_bytes, j)
            payload = bytes(model_bytes[k:k + ln])
            if field == 7:
                info = b"\x0a" + _varint(len(tensor_name)) + tensor_name.encode()
                payload += b"\x62" + _varint(len(info)) + info
                out += _varint(key) + _varint(len(payload)) + payload
            else:
                out += model_bytes[i:k + ln]
            i = k + ln
        elif wire == 5:
            out += model_bytes[i:j + 4]
            i = j + 4
        elif wire == 1:
            out += model_bytes[i:j + 8]
            i = j + 8
        else:
            raise ValueError(f"unexpected protobuf wire type {wire}")
    return bytes(out)


class Embedder:
    """MobileNetV2 pooled features -> unit-length float32 vector (1280)."""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, path):
        with open(path, "rb") as f:
            original = f.read()
        probe = np.zeros((64, 64, 3), np.uint8)
        self.net, self.out_name = None, None
        # OpenCV 4: the file as it is, pooled layer read by its layer name
        # ("onnx_node!GlobalAveragePool_97" in 4.8-4.10, plain name in older ones).
        # OpenCV 5: the same bytes plus one declared graph output (tensor "464").
        for data, name in ((original, "onnx_node!" + POOL_LAYER), (original, POOL_LAYER), (None, POOL_TENSOR)):
            try:
                if data is None:
                    data = add_graph_output(original, POOL_TENSOR)
                self.net = cv2.dnn.readNetFromONNX(np.frombuffer(data, np.uint8))
                self.out_name = name
                if self.embed(probe).shape == (EMBED_DIM,):
                    return
            except cv2.error:
                pass
        raise RuntimeError("could not read the pooled layer from the model")

    def embed(self, bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        self.net.setInput(np.ascontiguousarray(x))
        v = np.asarray(self.net.forward(self.out_name), np.float32).reshape(-1)
        return v / (np.linalg.norm(v) + 1e-9)

    @staticmethod
    def create(download=True):
        """Embedder, or None (with the reason printed) if the model can't be had."""
        try:
            path = ensure_model(download)
            if path is None:
                print("🏠 Room image model not downloaded yet; using corner features and landmarks only.")
                return None
            return Embedder(path)
        except Exception as e:
            print(f"🏠 Room image model off ({e!r}); using corner features and landmarks only.")
            return None


_orb = None


def orb_features(bgr):
    """(points Nx2 float32 in ORB_WIDTH-scaled pixels, descriptors Nx32 uint8)."""
    global _orb
    if _orb is None:
        _orb = cv2.ORB_create(nfeatures=ORB_FEATURES, fastThreshold=12)
    h, w = bgr.shape[:2]
    scale = ORB_WIDTH / float(w)
    gray = cv2.cvtColor(cv2.resize(bgr, (ORB_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    kps, des = _orb.detectAndCompute(gray, None)
    if des is None or not kps:
        return np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8)
    return np.array([k.pt for k in kps], np.float32), des


def sharpness(bgr):
    """Variance of the Laplacian on a small grey image: low = motion blur."""
    h, w = bgr.shape[:2]
    small = cv2.cvtColor(cv2.resize(bgr, (160, max(1, int(h * 160 / w)))), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def pack(embedding, points, descriptors):
    """One BLOB for room_photos.fingerprint (numpy .npz, no pickle)."""
    buf = io.BytesIO()
    np.savez(buf, emb=np.zeros(0, np.float16) if embedding is None else np.asarray(embedding, np.float16),
             pts=np.asarray(points, np.float32), des=np.asarray(descriptors, np.uint8))
    return buf.getvalue()


def unpack(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        emb = z["emb"].astype(np.float32)
        return (emb if emb.size else None), z["pts"], z["des"]
