import cv2
import pickle
import numpy as np
import pyttsx3
import threading
import queue
import time

from deepface import DeepFace


# ==========================================
# FACE DETECTOR
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==========================================
# LOAD DATABASE
# ==========================================

with open(
    "data/database/embeddings.pkl",
    "rb"
) as f:

    database = pickle.load(f)


# ==========================================
# SPEECH QUEUE
# ==========================================

speech_queue = queue.Queue()


def speech_worker():

    while True:

        message = speech_queue.get()

        if message is None:
            break

        print("VOICE:", message)

        try:

            # Create a completely fresh
            # TTS engine for every message

            engine = pyttsx3.init()

            engine.setProperty(
                "rate",
                150
            )

            engine.setProperty(
                "volume",
                1.0
            )

            engine.say(message)

            engine.runAndWait()

            engine.stop()

            del engine

        except Exception as e:

            print(
                "Speech error:",
                e
            )

        speech_queue.task_done()


speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)

speech_thread.start()


def speak(message):

    speech_queue.put(message)


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print(
        "ERROR: Could not access camera."
    )

    exit()


# ==========================================
# SETTINGS
# ==========================================

THRESHOLD = 0.30

RECOGNITION_INTERVAL = 10

STABLE_FRAMES = 2

NO_FACE_RESET_TIME = 1.5


# ==========================================
# STATE
# ==========================================

frame_count = 0

results = []


candidate_person = None

candidate_count = 0


current_person = None

last_face_time = time.time()


# ==========================================
# START
# ==========================================

print()
print("==============================")
print("SecondEye started")
print("==============================")
print("Press Q to quit.")
print()


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = camera.read()


    if not ret:

        print(
            "ERROR: Could not read camera."
        )

        break


    # ======================================
    # FACE DETECTION
    # ======================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70)
    )


    frame_count += 1


    # ======================================
    # RECOGNITION
    # ======================================

    if frame_count % RECOGNITION_INTERVAL == 0:

        results = []


        # ==================================
        # NO FACE
        # ==================================

        if len(faces) == 0:

            if current_person is not None:

                if (
                    time.time()
                    - last_face_time
                    > NO_FACE_RESET_TIME
                ):

                    print(
                        "No face detected."
                    )

                    current_person = None

                    candidate_person = None

                    candidate_count = 0


        # ==================================
        # FACE FOUND
        # ==================================

        else:

            last_face_time = time.time()


            # --------------------------------
            # Use largest face for now
            # --------------------------------

            largest_face = max(
                faces,
                key=lambda face:
                face[2] * face[3]
            )


            x, y, w, h = largest_face


            face_crop = frame[
                y:y+h,
                x:x+w
            ]


            try:

                # ==================================
                # CREATE EMBEDDING
                # ==================================

                embedding = DeepFace.represent(
                    img_path=face_crop,
                    model_name="Facenet512",
                    detector_backend="skip",
                    enforce_detection=False
                )[0]["embedding"]


                embedding = np.array(
                    embedding
                )


                # ==================================
                # FIND BEST MATCH
                # ==================================

                best_name = "Unknown"

                best_distance = float("inf")


                for entry in database:

                    stored = np.array(
                        entry["embedding"]
                    )


                    cosine_distance = 1 - (
                        np.dot(
                            embedding,
                            stored
                        )
                        /
                        (
                            np.linalg.norm(
                                embedding
                            )
                            *
                            np.linalg.norm(
                                stored
                            )
                        )
                    )


                    if cosine_distance < best_distance:

                        best_distance = (
                            cosine_distance
                        )

                        best_name = (
                            entry["name"]
                        )


                # ==================================
                # UNKNOWN THRESHOLD
                # ==================================

                if best_distance > THRESHOLD:

                    best_name = "Unknown"


                print(
                    f"Recognition: {best_name} "
                    f"| Distance: {best_distance:.2f} "
                    f"| Current: {current_person}"
                )


                # ==================================
                # STABILITY CHECK
                # ==================================

                if best_name == candidate_person:

                    candidate_count += 1

                else:

                    candidate_person = best_name

                    candidate_count = 1


                # ==================================
                # CONFIRM NEW PERSON
                # ==================================

                if (
                    candidate_count
                    >= STABLE_FRAMES
                    and
                    candidate_person
                    != current_person
                ):


                    current_person = (
                        candidate_person
                    )


                    print(
                        "PERSON CHANGED ->",
                        current_person
                    )


                    # ==================================
                    # SPEAK
                    # ==================================

                    if current_person == "Unknown":

                        message = (
                            "Unknown person detected."
                        )

                    else:

                        message = (
                            f"{current_person} detected."
                        )


                    print(
                        "ADDING TO SPEECH ->",
                        message
                    )


                    speak(message)


                # ==================================
                # SAVE RESULT
                # ==================================

                results.append(
                    (
                        x,
                        y,
                        w,
                        h,
                        best_name,
                        best_distance
                    )
                )


            except Exception as e:

                print(
                    "Recognition error:",
                    e
                )


    # ======================================
    # DRAW ALL FACE BOXES
    # ======================================

    for x, y, w, h in faces:

        name = "Unknown"

        distance = 1.0


        for (
            rx,
            ry,
            rw,
            rh,
            rname,
            rdistance
        ) in results:


            if (
                abs(x - rx) < 20
                and
                abs(y - ry) < 20
            ):

                name = rname

                distance = rdistance

                break


        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"{name} {distance:.2f}",
            (
                x,
                max(y - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ======================================
    # FACE COUNT
    # ======================================

    cv2.putText(
        frame,
        f"Faces detected: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ======================================
    # CURRENT PERSON
    # ======================================

    display_person = (
        current_person
        if current_person is not None
        else "No person"
    )


    cv2.putText(
        frame,
        f"Current: {display_person}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "SecondEye",
        frame
    )


    # ======================================
    # QUIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()

speech_queue.put(None)

print()
print("==============================")
print("SecondEye stopped.")
print("==============================")