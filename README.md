# SecondEye

**An experimental AI assistive-vision project that speaks what it sees.**

SecondEye is built as a "second pair of eyes" for blind and low-vision users. It recognises people, names objects and reads signs, then reports the results by voice, with direction (left, ahead, right) where it can.

> ⚠️ **Prototype, not a safety device.** SecondEye is a student project. It can miss obstacles, vehicles, kerbs and stairs, and its distance, face and text results can be wrong. It never tells you it is safe to cross a road. Do not use it in place of a cane, a guide dog or trained assistance.

---

## Two versions

| Version | What it is | Runs on |
|---|---|---|
| **Web demo** (`index.html`) | Hands-free live guide that runs in the browser | Any phone or laptop with a camera |
| **Python app** (`recognize.py`) | Desktop face-recognition app with voice announcements | Windows / macOS / Linux with a webcam |

🔗 **Live demo:** https://aaryamallik03.github.io/SecondEye/ *(requires HTTPS for camera access)*

---

## Web demo

Press **Start SecondEye** once and everything after that is spoken.

**Features**
- **People:** detects several faces at once and says how many people are in view. Faces you register are announced by name, others as "Unknown person".
- **Objects:** labels everyday objects (COCO-SSD, 80 classes) and says whether each is on your left, ahead or on your right, with a rough closeness estimate.
- **Vehicles and road signs:** gives priority warnings for cars, buses, trucks, motorcycles and bicycles, including "getting closer", and calls out stop signs. Traffic lights are reported with an estimated colour, never as "safe to cross".
- **Signs and text:** reads printed text from the live camera (Tesseract.js). Press **R** or the *Read sign / text* button for a careful on-demand scan.
- **Describe:** press **D** or the *Describe surroundings* button for a spoken summary of the scene.

**Privacy:** all processing happens in your browser and no video is uploaded. Faces you register are kept only for the current page session. AI model files are downloaded from a CDN when you start.

**Tech:** HTML/CSS/JavaScript, TensorFlow.js, COCO-SSD, face-api.js, Tesseract.js, Web Speech API.

**Run it locally**
```bash
# camera access needs HTTPS or localhost
python -m http.server 8000
# then open http://localhost:8000
```

**Tips for best results**
- Use the rear camera of a phone, held upright at chest height.
- Good lighting; keep still for a second or two.
- For text, hold large, high-contrast print steady so it fills a good part of the frame.

---

## Python desktop app

Recognises registered people through a webcam and announces them by voice.

**How it works**
1. **Detect** faces with an OpenCV Haar cascade.
2. **Embed** each face with DeepFace (Facenet512).
3. **Match** against registered people using cosine distance (threshold `0.30`). Anything above it is "Unknown".
4. **Stabilise** the result over consecutive checks, then **speak** it with `pyttsx3` ("Name detected." or "Unknown person detected.").

**Setup**
```bash
pip install opencv-python numpy pyttsx3 deepface
```

**Usage**
```bash
# 1. Register a person: type a name, press S to capture 10 photos (Q to cancel)
python register.py

# 2. Build the embeddings database from the registered photos
python create_embeddings.py

# 3. Start recognition (press Q to quit)
python recognize.py
```

Photos are stored in `data/faces/<name>/` and the embeddings database in `data/database/embeddings.pkl`.

---

## Project structure

```
SecondEye/
├── index.html              # Web demo
├── register.py             # Capture face photos for a new person
├── create_embeddings.py    # Build the embeddings database
├── recognize.py            # Real-time recognition with voice
├── app.py                  # Application entry point
├── test_database.py        # Tests
├── test_multiple_faces.py  # Tests
├── test_recognition.py     # Tests
├── test_voice.py           # Tests
└── data/                   # Registered faces and embeddings (local)
```

---

## Limitations

- The object detector knows only 80 common classes. Doors, stairs, kerbs, poles and holes are not among them.
- Distance is only a rough guess from object size.
- Face recognition is approximate and can fail in poor light or at an angle.
- Text reading works best on large, clear, well-lit print and struggles with glare, motion and distance.
- Traffic-light colour is a pixel estimate and can be wrong.

## Roadmap

- Stronger object detection (more classes, better accuracy)
- Voice commands, such as registering a person by speaking their name
- Better text reading for signs
- Testing with blind and low-vision users

## Author

**Aarya Mallik**: Computer Science Engineering, NIT Meghalaya

## Disclaimer

This project is for research and learning. It is not certified for safety-critical use.
