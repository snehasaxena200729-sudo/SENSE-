import ssl
# Tell Mac: "It's safe to download the tiny Whisper model from OpenAI"
ssl._create_default_https_context = ssl._create_unverified_context

import speech_recognition as sr
import time

recognizer = sr.Recognizer()

print("-------------------------------------------------------")
print("SENSE Offline Voice Engine (Zero Cloud / 100% Local)")
print("3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1... SPEAK NOW! (e.g. say 'Where is my phone')")
print("-------------------------------------------------------")

with sr.Microphone() as source:
    recognizer.adjust_for_ambient_noise(source, duration=0.8)
    audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
    print("🧠 Transcribing locally on your Mac...")

try:
    text = recognizer.recognize_whisper(audio, model="tiny.en")
    print(f"\n🎉 SUCCESS! SENSE heard: '{text.strip()}'")
except Exception as e:
    print(f"\nError: {e}")