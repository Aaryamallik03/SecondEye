import cv2
import numpy as np
import pickle
import threading
import queue
import time
import subprocess
from deepface import DeepFace
import easyocr

THRESHOLD = 0.30
RECOGNITION_INTERVAL = 5

NO_FACE_RESET_TIME = 4.0
MAX_MATCH_DISTANCE = 180

UNKNOWN_CONFIRM_FRAMES = 3
KNOWN_CONFIRM_FRAMES = 3
POSITION_CONFIRM_FRAMES = 3

LEFT_ENTER = 0.32
LEFT_EXIT = 0.40

RIGHT_ENTER = 0.68
RIGHT_EXIT = 0.60

OCR_MIN_CONFIDENCE = 0.40

with open("data/database/embeddings.pkl", "rb") as f:
    database = pickle.load(f)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# --------------------------------------------------------------------
# TEXT READING (OCR)
#
# Loading EasyOCR's model takes a few seconds - this happens once at
# startup, not per read, so pressing 'T' during use is fast. Runs on
# CPU by default (gpu=False); set gpu=True if you have a CUDA GPU set
# up and want faster reads.
# --------------------------------------------------------------------

print("Loading text reader (first run downloads the OCR model)...")
ocr_reader = easyocr.Reader(["en"], gpu=False)
print("Text reader ready.")

ocr_queue = queue.Queue()


def ocr_worker():
    while True:
        frame_to_read = ocr_queue.get()

        if frame_to_read is None:
            break

        try:
            results = ocr_reader.readtext(frame_to_read)

            texts = [
                text for (_, text, confidence) in results
                if confidence > OCR_MIN_CONFIDENCE
            ]

            if texts:
                combined = ". ".join(texts)
                print("TEXT READ:", combined)
                speak(f"Text reads: {combined}")
            else:
                print("TEXT READ: nothing found")
                speak("No readable text found.")

        except Exception as e:
            print("OCR ERROR:", e)

        finally:
            ocr_queue.task_done()


ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
ocr_thread.start()


speech_queue = queue.Queue()


def speech_worker():
    while True:
        text = speech_queue.get()

        if text is None:
            break

        try:
            print("VOICE:", text)

            safe_text = text.replace("'", "''")

            command = (
                "Add-Type -AssemblyName System.Speech; "
                "$speaker = New-Object "
                "System.Speech.Synthesis.SpeechSynthesizer; "
                "$speaker.Rate = 0; "
                "$speaker.Volume = 100; "
                f"$speaker.Speak('{safe_text}'); "
                "$speaker.Dispose();"
            )

            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    command
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        except Exception as e:
            print("TTS ERROR:", e)

        finally:
            speech_queue.task_done()


speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)

speech_thread.start()


def speak(text):
    speech_queue.put(text)


def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)

    return 1 - (
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )
    )


def get_position(center_x, frame_width, previous_position):
    ratio = center_x / frame_width

    if previous_position == "left":
        if ratio > LEFT_EXIT:
            return "right" if ratio > RIGHT_ENTER else "center"
        return "left"

    if previous_position == "right":
        if ratio < RIGHT_EXIT:
            return "left" if ratio < LEFT_ENTER else "center"
        return "right"

    if ratio < LEFT_ENTER:
        return "left"

    if ratio > RIGHT_ENTER:
        return "right"

    return "center"


def assign_tracks(detected_faces, previous_faces, max_distance):
    candidates = []

    for face_idx, face in enumerate(detected_faces):
        center = np.array(face["center"])

        for person_id, previous_center in previous_faces.items():
            distance = np.linalg.norm(center - np.array(previous_center))

            if distance < max_distance:
                candidates.append((distance, face_idx, person_id))

    candidates.sort(key=lambda c: c[0])

    assigned_face_idx = set()
    assigned_person_id = set()
    face_to_person = {}

    for distance, face_idx, person_id in candidates:
        if face_idx in assigned_face_idx:
            continue
        if person_id in assigned_person_id:
            continue

        face_to_person[face_idx] = person_id
        assigned_face_idx.add(face_idx)
        assigned_person_id.add(person_id)

    return face_to_person, assigned_face_idx


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()


current_people = {}
candidate_people = {}
candidate_counts = {}

previous_faces = {}
last_seen = {}

next_person_id = 0

position_candidates = {}
position_counts = {}
confirmed_positions = {}

frame_count = 0
last_face_time = time.time()

text_reading_flash_until = 0.0


print()
print("==============================")
print("SecondEye - Persistent Tracking")
print("==============================")
print("Press Q to quit.")
print("Press T to read text in view aloud.")
print()


def handle_key(key, frame):
    """
    Shared key handler used in both the no-face branch and the main
    branch, so 'T' works no matter what's happening in the frame.
    Returns True if the program should quit.

    Accepts both upper and lower case, since Caps Lock (or Shift)
    being on would otherwise make T/Q silently do nothing - cv2's
    waitKey() returns the actual key code, uppercase and lowercase
    are different codes.
    """

    global text_reading_flash_until

    if key in (ord("q"), ord("Q")):
        return True

    if key in (ord("t"), ord("T")):
        print("Reading text in view...")
        speak("Reading text.")
        ocr_queue.put(frame.copy())
        text_reading_flash_until = time.time() + 1.0

    return False


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(80, 80)
    )

    # ------------------------------------------------
    # NO FACE DETECTED
    # ------------------------------------------------

    if len(faces) == 0:

        print("No faces detected.")

        if time.time() - last_face_time > NO_FACE_RESET_TIME:

            expired_ids = [
                person_id
                for person_id in previous_faces
                if time.time() - last_seen.get(person_id, 0) > NO_FACE_RESET_TIME
            ]

            for person_id in expired_ids:
                previous_faces.pop(person_id, None)
                last_seen.pop(person_id, None)
                current_people.pop(person_id, None)
                candidate_people.pop(person_id, None)
                candidate_counts.pop(person_id, None)
                position_candidates.pop(person_id, None)
                position_counts.pop(person_id, None)
                confirmed_positions.pop(person_id, None)

        if time.time() < text_reading_flash_until:
            cv2.putText(
                frame, "READING TEXT...", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2
            )

        cv2.imshow("SecondEye", frame)

        key = cv2.waitKey(1) & 0xFF
        if handle_key(key, frame):
            break

        continue

    last_face_time = time.time()

    # ------------------------------------------------
    # CREATE DETECTED FACE DATA
    # ------------------------------------------------

    detected_faces = []

    for (x, y, w, h) in faces:
        center_x = x + w // 2
        center_y = y + h // 2

        detected_faces.append({
            "box": (x, y, w, h),
            "center": (center_x, center_y)
        })

    # ------------------------------------------------
    # TRACK FACES
    # ------------------------------------------------

    face_to_person, assigned_face_idx = assign_tracks(
        detected_faces,
        previous_faces,
        MAX_MATCH_DISTANCE
    )

    current_frame_people = {}

    for face_idx, face in enumerate(detected_faces):

        if face_idx in face_to_person:
            person_id = face_to_person[face_idx]

        else:
            person_id = next_person_id
            next_person_id += 1

            print(f"NEW TRACK -> Person ID {person_id}")

        current_frame_people[person_id] = face
        last_seen[person_id] = time.time()

    for person_id, face_data in current_frame_people.items():
        previous_faces[person_id] = face_data["center"]

    # ------------------------------------------------
    # RECOGNITION
    # ------------------------------------------------

    if frame_count % RECOGNITION_INTERVAL == 0:

        for person_id, face_data in current_frame_people.items():

            x, y, w, h = face_data["box"]
            face_crop = frame[y:y + h, x:x + w]

            if face_crop.size == 0:
                continue

            try:
                result = DeepFace.represent(
                    img_path=face_crop,
                    model_name="Facenet512",
                    detector_backend="skip",
                    enforce_detection=False
                )

                embedding = result[0]["embedding"]

                best_name = "Unknown"
                best_distance = 999

                for entry in database:
                    distance = cosine_distance(embedding, entry["embedding"])

                    if distance < best_distance:
                        best_distance = distance
                        best_name = entry["name"]

                if best_distance > THRESHOLD:
                    best_name = "Unknown"

                print(
                    f"Person ID {person_id}: "
                    f"{best_name} | Distance: {best_distance:.2f}"
                )

                current_identity = current_people.get(person_id)

                if person_id not in current_people:
                    candidate_people[person_id] = best_name
                    candidate_counts[person_id] = 1
                    current_people[person_id] = None

                elif current_identity == best_name:
                    candidate_counts[person_id] = 0

                else:
                    if candidate_people.get(person_id) == best_name:
                        candidate_counts[person_id] = candidate_counts.get(person_id, 0) + 1
                    else:
                        candidate_people[person_id] = best_name
                        candidate_counts[person_id] = 1

                    required_frames = (
                        UNKNOWN_CONFIRM_FRAMES
                        if best_name == "Unknown"
                        else KNOWN_CONFIRM_FRAMES
                    )

                    if candidate_counts.get(person_id, 0) >= required_frames:

                        current_people[person_id] = best_name
                        candidate_counts[person_id] = 0

                        print(f"PERSON CHANGED -> {best_name}")

                        position = get_position(
                            face_data["center"][0],
                            frame_width,
                            confirmed_positions.get(person_id, "center")
                        )

                        if best_name == "Unknown":
                            if position == "left":
                                speak("Unknown person detected, on your left.")
                            elif position == "right":
                                speak("Unknown person detected, on your right.")
                            else:
                                speak("Unknown person detected, in front of you.")

                        else:
                            if position == "left":
                                speak(f"{best_name} detected, slightly to your left.")
                            elif position == "right":
                                speak(f"{best_name} detected, slightly to your right.")
                            else:
                                speak(f"{best_name} detected, in front of you.")

            except Exception as e:
                print(f"Recognition error for Person ID {person_id}:", e)

    # ------------------------------------------------
    # POSITION TRACKING
    # ------------------------------------------------

    for person_id, face_data in current_frame_people.items():

        if current_people.get(person_id) is None:
            continue

        center_x = face_data["center"][0]

        previous_confirmed = confirmed_positions.get(person_id, "center")

        new_position = get_position(center_x, frame_width, previous_confirmed)

        old_candidate = position_candidates.get(person_id)

        if old_candidate == new_position:
            position_counts[person_id] = position_counts.get(person_id, 0) + 1
        else:
            position_candidates[person_id] = new_position
            position_counts[person_id] = 1

        if position_counts.get(person_id, 0) >= POSITION_CONFIRM_FRAMES:

            confirmed_position = position_candidates[person_id]
            previous_position = confirmed_positions.get(person_id)

            if previous_position != confirmed_position:

                confirmed_positions[person_id] = confirmed_position
                identity = current_people.get(person_id)

                if previous_position is not None:

                    if identity == "Unknown":
                        if confirmed_position == "left":
                            speak("Unknown person moved to your left.")
                        elif confirmed_position == "right":
                            speak("Unknown person moved to your right.")
                        else:
                            speak("Unknown person is in front of you.")

                    else:
                        if confirmed_position == "left":
                            speak(f"{identity} is slightly to your left.")
                        elif confirmed_position == "right":
                            speak(f"{identity} is slightly to your right.")
                        else:
                            speak(f"{identity} is in front of you.")

            position_counts[person_id] = 0

    # ------------------------------------------------
    # DRAW RESULTS
    # ------------------------------------------------

    for person_id, face_data in current_frame_people.items():

        x, y, w, h = face_data["box"]
        identity = current_people.get(person_id, "...")
        position = confirmed_positions.get(person_id, "...")

        label = f"ID {person_id}: {identity} | {position}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    left_boundary = int(frame_width * LEFT_ENTER)
    right_boundary = int(frame_width * RIGHT_ENTER)

    cv2.line(frame, (left_boundary, 0), (left_boundary, frame_height), (255, 255, 0), 1)
    cv2.line(frame, (right_boundary, 0), (right_boundary, frame_height), (255, 255, 0), 1)

    cv2.putText(frame, "LEFT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(
        frame, "CENTER", (int(frame_width * 0.43), 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )
    cv2.putText(
        frame, "RIGHT", (int(frame_width * 0.85), 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )

    if time.time() < text_reading_flash_until:
        cv2.putText(
            frame, "READING TEXT...", (20, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2
        )

    cv2.imshow("SecondEye", frame)

    key = cv2.waitKey(1) & 0xFF
    if handle_key(key, frame):
        break

    frame_count += 1


cap.release()
cv2.destroyAllWindows()

speech_queue.put(None)
ocr_queue.put(None)

print()
print("SecondEye stopped.")