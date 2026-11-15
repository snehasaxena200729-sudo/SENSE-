import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import numpy as np
import time
import subprocess
import threading
from collections import defaultdict, Counter
import speech_recognition as sr
from ultralytics import YOLO
import memory_db

# ===============================