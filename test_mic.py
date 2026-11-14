import speech_recognition as sr
import time

recognizer = sr.Recognizer()

print("-----------------------------------------")
print("Getting ready... Speak clearly into your laptop.")
print("3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1... SPEAK NOW! (e.g. say 'Where is my phone')")
print("-----------------------------------------")

with sr.Microphone() as source:
    recognizer.adjust_for_ambient_noise(source, duration=0.8)
    audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
    print("Processing what you said...")

try:
    text = recognizer.recognize_google(audio)
    print(f"\n🎉 SUCCESS! SENSE heard you say: '{text}'")
except Exception as e:
    print(f"\nError: {e}")