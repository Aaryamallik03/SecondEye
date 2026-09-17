import cv2
import numpy as np
import pickle
import threading
import queue
import time
import subprocess
from deepface import DeepFace

THRESHOLD = 0.30
RECOGNITION_INTERVAL = 10

NO_FACE_RESET_TIME = 4.0
MAX_MATCH_DISTANCE = 180

UNKNOWN_CONFIRM_FRAMES = 3
KNOWN_CONFIRM_FRAMES = 3
POSITION_CONFIRM_FRAMES = 3

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


def get_position(center_x, frame_width):
    ratio = center_x / frame_width

    if ratio < 0.35:
        return "left"

    elif ratio > 0.65:
        return "right"

    else:
        return "center"


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

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )


    # ------------------------------------------------
    # NO FACE DETECTED
    # ------------------------------------------------

    if len(faces) == 0:

        print("No faces detected.")

        # IMPORTANT:
        # We do NOT immediately delete tracking information.
        # This allows a person to disappear briefly and return
        # without receiving a new ID.

        if time.time() - last_face_time > NO_FACE_RESET_TIME:

            expired_ids = []

            for person_id in previous_faces:

                if (
                    time.time()
                    -
                    last_seen.get(person_id, 0)
                    >
                    NO_FACE_RESET_TIME
                ):
                    expired_ids.append(person_id)


            for person_id in expired_ids:

                previous_faces.pop(
                    person_id,
                    None
                )

                last_seen.pop(
                    person_id,
                    None
                )

                current_people.pop(
                    person_id,
                    None
                )

                candidate_people.pop(
                    person_id,
                    None
                )

                candidate_counts.pop(
                    person_id,
                    None
                )

                position_candidates.pop(
                    person_id,
                    None
                )

                position_counts.pop(
                    person_id,
                    None
                )

                confirmed_positions.pop(
                    person_id,
                    None
                )


        cv2.imshow(
            "SecondEye",
            frame
        )

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
    # TRACK FACES
    # ------------------------------------------------

    matched_ids = set()

    current_frame_people = {}


    for face in detected_faces:

        center = face["center"]

        best_id = None
        best_distance = MAX_MATCH_DISTANCE


        for person_id, previous_center in previous_faces.items():

            if person_id in matched_ids:
                continue


            distance = np.linalg.norm(
                np.array(center)
                -
                np.array(previous_center)
            )


            if distance < best_distance:

                best_distance = distance
                best_id = person_id


        # If no existing person is close enough,
        # create a new ID.

        if best_id is None:

            best_id = next_person_id
            next_person_id += 1

            print(
                f"NEW TRACK -> Person ID {best_id}"
            )


        matched_ids.add(best_id)

        current_frame_people[
            best_id
        ] = face

        last_seen[
            best_id
        ] = time.time()


    # ------------------------------------------------
    # UPDATE TRACKING CENTERS
    # ------------------------------------------------

    for person_id, face_data in current_frame_people.items():

        previous_faces[
            person_id
        ] = face_data["center"]


    # ------------------------------------------------
    # RECOGNITION
    # ------------------------------------------------

    if frame_count % RECOGNITION_INTERVAL == 0:

        for person_id, face_data in current_frame_people.items():

            x, y, w, h = face_data["box"]

            face_crop = frame[
                y:y + h,
                x:x + w
            ]


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

                    distance = cosine_distance(
                        embedding,
                        entry["embedding"]
                    )


                    if distance < best_distance:

                        best_distance = distance
                        best_name = entry["name"]


                if best_distance > THRESHOLD:

                    best_name = "Unknown"


                print(
                    f"Person ID {person_id}: "
                    f"{best_name} | Distance: "
                    f"{best_distance:.2f}"
                )


                # ------------------------------------------------
                # FIRST RECOGNITION
                # ------------------------------------------------

                if person_id not in current_people:

                    candidate_people[
                        person_id
                    ] = best_name

                    candidate_counts[
                        person_id
                    ] = 1

                    current_people[
                        person_id
                    ] = None


                else:

                    current_identity = current_people[
                        person_id
                    ]


                    # ------------------------------------------------
                    # IDENTITY NOT YET CONFIRMED
                    # ------------------------------------------------

                    if current_identity is None:

                        if (
                            candidate_people.get(person_id)
                            ==
                            best_name
                        ):

                            candidate_counts[
                                person_id
                            ] += 1

                        else:

                            candidate_people[
                                person_id
                            ] = best_name

                            candidate_counts[
                                person_id
                            ] = 1


                        required_frames = (
                            UNKNOWN_CONFIRM_FRAMES
                            if best_name == "Unknown"
                            else KNOWN_CONFIRM_FRAMES
                        )


                        if (
                            candidate_counts.get(person_id, 0)
                            >=
                            required_frames
                        ):

                            current_people[
                                person_id
                            ] = best_name

                            candidate_counts[
                                person_id
                            ] = 0


                            print(
                                "PERSON CHANGED -> "
                                f"{best_name}"
                            )


                            position = get_position(
                                face_data["center"][0],
                                frame_width
                            )


                            if best_name == "Unknown":

                                if position == "left":

                                    speak(
                                        "Unknown person detected, "
                                        "on your left."
                                    )

                                elif position == "right":

                                    speak(
                                        "Unknown person detected, "
                                        "on your right."
                                    )

                                else:

                                    speak(
                                        "Unknown person detected, "
                                        "in front of you."
                                    )

                            else:

                                if position == "left":

                                    speak(
                                        f"{best_name} detected, "
                                        "slightly to your left."
                                    )

                                elif position == "right":

                                    speak(
                                        f"{best_name} detected, "
                                        "slightly to your right."
                                    )

                                else:

                                    speak(
                                        f"{best_name} detected, "
                                        "in front of you."
                                    )


                    # ------------------------------------------------
                    # SAME IDENTITY
                    # ------------------------------------------------

                    elif best_name == current_identity:

                        candidate_counts[
                            person_id
                        ] = 0


                    # ------------------------------------------------
                    # POSSIBLE IDENTITY CHANGE
                    # ------------------------------------------------

                    else:

                        if (
                            candidate_people.get(person_id)
                            ==
                            best_name
                        ):

                            candidate_counts[
                                person_id
                            ] += 1

                        else:

                            candidate_people[
                                person_id
                            ] = best_name

                            candidate_counts[
                                person_id
                            ] = 1


                        required_frames = (
                            UNKNOWN_CONFIRM_FRAMES
                            if best_name == "Unknown"
                            else KNOWN_CONFIRM_FRAMES
                        )


                        if (
                            candidate_counts.get(person_id, 0)
                            >=
                            required_frames
                        ):

                            current_people[
                                person_id
                            ] = best_name

                            candidate_counts[
                                person_id
                            ] = 0


                            print(
                                "PERSON CHANGED -> "
                                f"{best_name}"
                            )


                            position = get_position(
                                face_data["center"][0],
                                frame_width
                            )


                            if best_name == "Unknown":

                                if position == "left":

                                    speak(
                                        "Unknown person detected, "
                                        "on your left."
                                    )

                                elif position == "right":

                                    speak(
                                        "Unknown person detected, "
                                        "on your right."
                                    )

                                else:

                                    speak(
                                        "Unknown person detected, "
                                        "in front of you."
                                    )

                            else:

                                if position == "left":

                                    speak(
                                        f"{best_name} detected, "
                                        "slightly to your left."
                                    )

                                elif position == "right":

                                    speak(
                                        f"{best_name} detected, "
                                        "slightly to your right."
                                    )

                                else:

                                    speak(
                                        f"{best_name} detected, "
                                        "in front of you."
                                    )


            except Exception as e:

                print(
                    "Recognition error "
                    f"for Person ID {person_id}:",
                    e
                )


    # ------------------------------------------------
    # POSITION TRACKING
    # ------------------------------------------------

    for person_id, face_data in current_frame_people.items():

        if current_people.get(person_id) is None:
            continue


        center_x = face_data["center"][0]

        new_position = get_position(
            center_x,
            frame_width
        )


        old_candidate = position_candidates.get(
            person_id
        )


        if old_candidate == new_position:

            position_counts[
                person_id
            ] = position_counts.get(
                person_id,
                0
            ) + 1

        else:

            position_candidates[
                person_id
            ] = new_position

            position_counts[
                person_id
            ] = 1


        if (
            position_counts.get(person_id, 0)
            >=
            POSITION_CONFIRM_FRAMES
        ):

            confirmed_position = position_candidates[
                person_id
            ]


            previous_position = confirmed_positions.get(
                person_id
            )


            if previous_position != confirmed_position:

                confirmed_positions[
                    person_id
                ] = confirmed_position


                identity = current_people.get(
                    person_id
                )


                # Only announce movement after
                # the person's position has changed.

                if previous_position is not None:

                    if identity == "Unknown":

                        if confirmed_position == "left":

                            speak(
                                "Unknown person "
                                "moved to your left."
                            )

                        elif confirmed_position == "right":

                            speak(
                                "Unknown person "
                                "moved to your right."
                            )

                        else:

                            speak(
                                "Unknown person "
                                "is in front of you."
                            )

                    else:

                        if confirmed_position == "left":

                            speak(
                                f"{identity} is "
                                "slightly to your left."
                            )

                        elif confirmed_position == "right":

                            speak(
                                f"{identity} is "
                                "slightly to your right."
                            )

                        else:

                            speak(
                                f"{identity} is "
                                "in front of you."
                            )


            position_counts[
                person_id
            ] = 0


    # ------------------------------------------------
    # DRAW RESULTS
    # ------------------------------------------------

    for person_id, face_data in current_frame_people.items():

        x, y, w, h = face_data["box"]


        identity = current_people.get(
            person_id,
            "..."
        )


        position = confirmed_positions.get(
            person_id,
            "..."
        )


        label = (
            f"ID {person_id}: "
            f"{identity} | "
            f"{position}"
        )


        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # ------------------------------------------------
    # POSITION BOUNDARIES
    # ------------------------------------------------

    left_boundary = int(
        frame_width * 0.35
    )

    right_boundary = int(
        frame_width * 0.65
    )


    cv2.line(
        frame,
        (left_boundary, 0),
        (left_boundary, frame_height),
        (255, 255, 0),
        1
    )


    cv2.line(
        frame,
        (right_boundary, 0),
        (right_boundary, frame_height),
        (255, 255, 0),
        1
    )


    cv2.putText(
        frame,
        "LEFT",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "CENTER",
        (
            int(frame_width * 0.43),
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "RIGHT",
        (
            int(frame_width * 0.85),
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    cv2.imshow(
        "SecondEye",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


    frame_count += 1


cap.release()

cv2.destroyAllWindows()

speech_queue.put(None)

print()
print("SecondEye stopped.")