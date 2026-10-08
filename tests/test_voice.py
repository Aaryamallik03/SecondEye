import pyttsx3

engine = pyttsx3.init()

engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

print("Speaking unknown person...")

engine.say("Unknown person detected.")
engine.runAndWait()

print("Finished.")