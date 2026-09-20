import cv2
import numpy as np
import pickle
import threading
import queue
import time
import subprocess
from deepface import DeepFace

THRESHOLD = 0.30
RECOGNITION_INTERVAL = 5

NO_FACE_RESET_TIME = 4.0
MAX_MATCH_DISTANCE = 180

UNKNOWN_CONFIRM_FRAMES = 3
KNOWN_CONFIRM_FRAMES = 3
POSITION_CONFIRM_FRAMES = 3

# --------------------------------------------------------------------
# POSITION HYSTERESIS
#
# The old code used a single pair of boundaries (0.35 / 0.65) with no
# dead zone. A face sitting right at ratio ~0.35 can jitter a few
# pixels frame to frame and cross that line repeatedly, which reset
# the POSITION_CONFIRM_FRAMES counter every time and occasionally let
# 3 "wrong side" readings land in a row - that's the flicker.
#
# Fix: use a WIDER exit threshold than the entry threshold, based on
# the position the track is already confirmed to be in. To leave
# "left", you now have to move further right than what it took to
# enter "left" in the first place. This creates a genuine dead zone
# instead of a single hair-trigger line.
# --------------------------------------------------------------------

LEFT_ENTER = 0.32
LEFT_EXIT = 0.40

RIGHT_ENTER = 0.68
RIGHT_EXIT = 0.60

with open("data/database/embeddings.pkl", "rb") as f:
    database = pickle.load(f)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

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
    """
    Returns left / center / right, but with hysteresis: the threshold
    to LEAVE a zone is further than the threshold that was needed to
    ENTER it. This stops small jitter near a boundary from flipping
    the reading back and forth.
    """

    ratio = center_x / frame_width

    if previous_position == "left":
        if ratio > LEFT_EXIT:
            return "right" if ratio > RIGHT_ENTER else "center"
        return "left"

    if previous_position == "right":
        if ratio < RIGHT_EXIT:
            return "left" if ratio < LEFT_ENTER else "center"
        return "right"

    # previous_position == "center" or unknown - use the tighter
    # entry thresholds to decide if we've moved into left/right.

    if ratio < LEFT_ENTER:
        return "left"

    if ratio > RIGHT_ENTER:
        return "right"

    return "center"


def assign_tracks(detected_faces, previous_faces, max_distance):
    """
    Global greedy nearest-neighbor assignment.

    The old version looped through detected_faces in whatever order
    the cascade returned them, and for each face grabbed whichever
    previous track was closest AT THAT POINT. That's a per-face
    greedy match, not a global one - so when two people are close
    together, the face processed first can steal the track that
    actually belonged to the second face, and IDs (and therefore
    names) swap between people.

    Fix: build every (distance, face_index, person_id) pair, sort ALL
    of them by distance ascending, and assign in that order, skipping
    any face or person_id that's already been claimed. This is a
    standard greedy approximation of optimal assignment and is a big
    step up from order-dependent matching, without needing a full
    Hungarian-algorithm dependency.
    """

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


print()
print("==============================")
print("SecondEye - Persistent Tracking")
print("==============================")
print("Press Q to quit.")
print()


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

        cv2.imshow("SecondEye", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
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
    # TRACK FACES (global assignment, see assign_tracks())
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

    # ------------------------------------------------
    # UPDATE TRACKING CENTERS
    # ------------------------------------------------

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
                    # Identity confirmed and matches again - lock it
                    # in, reset the candidate counter. This is what
                    # keeps a confirmed identity from being nudged
                    # away by a single noisy embedding.
                    candidate_counts[person_id] = 0

                else:
                    # Either identity not yet confirmed, or this is a
                    # possible change away from a confirmed identity.
                    # Both cases use the same confirm-over-N-frames
                    # logic before accepting the new name.

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
    # POSITION TRACKING (with hysteresis)
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

    # ------------------------------------------------
    # POSITION BOUNDARIES (visual reference only - the real
    # hysteresis logic lives in get_position(), these lines just
    # show the entry thresholds for a rough visual guide)
    # ------------------------------------------------

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

    cv2.imshow("SecondEye", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1


cap.release()
cv2.destroyAllWindows()
speech_queue.put(None)

print()
print("SecondEye stopped.")