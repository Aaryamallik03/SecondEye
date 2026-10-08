import cv2
import numpy as np
import pickle
import threading
import queue
from collections import Counter

from speech_manager import SpeechManager
from deepface import DeepFace
import easyocr


# =========================
# CONFIGURATION
# =========================

THRESHOLD = 0.30

RECOGNITION_INTERVAL = 5

MAX_MATCH_DISTANCE = 180
MAX_MISSED_FRAMES = 15

IDENTITY_HISTORY_SIZE = 5
MIN_IDENTITY_VOTES = 3

POSITION_CONFIRM_FRAMES = 5

# Position boundaries
LEFT_ENTER = 0.35
LEFT_EXIT = 0.40

RIGHT_ENTER = 0.65
RIGHT_EXIT = 0.60

OCR_MIN_CONFIDENCE = 0.40

SPEECH_COOLDOWN = 4.0


# =========================
# SPEECH MANAGER
# =========================

speech_manager = SpeechManager(
    cooldown=SPEECH_COOLDOWN
)


# =========================
# LOAD FACE DATABASE
# =========================

with open(
    "data/database/embeddings.pkl",
    "rb"
) as f:

    face_database = pickle.load(f)


print(
    "Embedding database type:",
    type(face_database).__name__
)

print(
    "Number of stored embeddings:",
    len(face_database)
)


# =========================
# DATABASE NORMALIZATION
# =========================

def get_database_entries(database):

    """
    Converts the existing embedding database into:

        [
            (identity, embedding),
            ...
        ]

    Supports common list/dictionary formats.
    """

    entries = []

    # -------------------------
    # Dictionary database
    # -------------------------

    if isinstance(database, dict):

        for identity, value in database.items():

            # Direct embedding
            if isinstance(value, (list, tuple, np.ndarray)):

                entries.append(
                    (
                        identity,
                        np.asarray(value)
                    )
                )

            # Dictionary containing embedding
            elif isinstance(value, dict):

                embedding = (
                    value.get("embedding")
                    or value.get("embeddings")
                )

                if embedding is not None:

                    entries.append(
                        (
                            identity,
                            np.asarray(embedding)
                        )
                    )

        return entries

    # -------------------------
    # List database
    # -------------------------

    if isinstance(database, list):

        for item in database:

            # -------------------------
            # Dictionary item
            # -------------------------

            if isinstance(item, dict):

                identity = (
                    item.get("name")
                    or item.get("identity")
                    or item.get("person")
                    or item.get("label")
                )

                embedding = (
                    item.get("embedding")
                    or item.get("embeddings")
                )

                if (
                    identity is not None
                    and embedding is not None
                ):

                    entries.append(
                        (
                            identity,
                            np.asarray(embedding)
                        )
                    )

                    continue

            # -------------------------
            # Tuple/list:
            # (identity, embedding)
            # -------------------------

            if isinstance(item, (list, tuple)):

                if len(item) >= 2:

                    identity = item[0]
                    embedding = item[1]

                    if isinstance(
                        identity,
                        str
                    ):

                        entries.append(
                            (
                                identity,
                                np.asarray(embedding)
                            )
                        )

        return entries

    return entries


database_entries = get_database_entries(
    face_database
)


if len(database_entries) == 0:

    print()
    print(
        "WARNING: No valid face embeddings were found."
    )
    print(
        "Check data/database/embeddings.pkl"
    )
    print()

else:

    print(
        "Loaded identities:",
        [
            identity
            for identity, _ in database_entries
        ]
    )


# =========================
# FACE DETECTOR
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


# =========================
# OCR
# =========================

print("Loading OCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=False
)

print("OCR ready.")


ocr_queue = queue.Queue()


def ocr_worker():

    while True:

        frame = ocr_queue.get()

        if frame is None:

            ocr_queue.task_done()

            break

        try:

            results = reader.readtext(frame)

            detected_text = []

            for _, text, confidence in results:

                if confidence >= OCR_MIN_CONFIDENCE:

                    detected_text.append(text)

            if detected_text:

                final_text = " ".join(
                    detected_text
                )

                print(
                    "OCR:",
                    final_text
                )

                speech_manager.announce_text(
                    f"I can read: {final_text}"
                )

        except Exception as e:

            print(
                "OCR error:",
                e
            )

        finally:

            ocr_queue.task_done()


ocr_thread = threading.Thread(
    target=ocr_worker,
    daemon=True
)

ocr_thread.start()


# =========================
# UTILITY FUNCTIONS
# =========================

def cosine_distance(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:

        return float("inf")

    return 1 - (
        np.dot(a, b)
        / denominator
    )


def calculate_iou(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)

    xb = min(
        x1 + w1,
        x2 + w2
    )

    yb = min(
        y1 + h1,
        y2 + h2
    )

    intersection_width = max(
        0,
        xb - xa
    )

    intersection_height = max(
        0,
        yb - ya
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area1 = w1 * h1
    area2 = w2 * h2

    union_area = (
        area1
        + area2
        - intersection_area
    )

    if union_area == 0:

        return 0

    return (
        intersection_area
        / union_area
    )


# =========================
# POSITION DETECTION
# =========================

def get_position(
    center_x,
    frame_width,
    previous_position
):

    relative_x = (
        center_x / frame_width
    )

    # -------------------------
    # Currently LEFT
    # -------------------------

    if previous_position == "left":

        if relative_x < LEFT_EXIT:

            return "left"

        if relative_x < RIGHT_EXIT:

            return "center"

        return "right"

    # -------------------------
    # Currently RIGHT
    # -------------------------

    if previous_position == "right":

        if relative_x > RIGHT_EXIT:

            return "right"

        if relative_x > LEFT_EXIT:

            return "center"

        return "left"

    # -------------------------
    # Currently CENTER
    # -------------------------

    if relative_x < LEFT_ENTER:

        return "left"

    if relative_x > RIGHT_ENTER:

        return "right"

    return "center"


# =========================
# FACE TRACK
# =========================

class FaceTrack:

    def __init__(
        self,
        track_id,
        center,
        box
    ):

        self.track_id = track_id

        self.center = center
        self.box = box

        # Identity
        self.identity = None
        self.identity_distance = None

        self.identity_history = []

        # Tracking
        self.missed_frames = 0

        # Position
        self.position = "center"

        self.position_candidate = None
        self.position_count = 0

        # Speech
        self.has_been_announced = False

    # -------------------------
    # Update
    # -------------------------

    def update(
        self,
        center,
        box
    ):

        self.center = center
        self.box = box

        self.missed_frames = 0

    # -------------------------
    # Recognition history
    # -------------------------

    def add_recognition(
        self,
        identity,
        distance
    ):

        self.identity_history.append(
            identity
        )

        if len(
            self.identity_history
        ) > IDENTITY_HISTORY_SIZE:

            self.identity_history.pop(0)

        self.identity_distance = distance

        counts = Counter(
            self.identity_history
        )

        identity_name, votes = (
            counts.most_common(1)[0]
        )

        if votes >= MIN_IDENTITY_VOTES:

            old_identity = self.identity

            self.identity = identity_name

            return (
                True,
                old_identity
            )

        return (
            False,
            self.identity
        )

    # -------------------------
    # Position
    # -------------------------

    def update_position(
        self,
        frame_width
    ):

        new_position = get_position(
            self.center[0],
            frame_width,
            self.position
        )

        # Same position
        if new_position == self.position:

            self.position_candidate = None
            self.position_count = 0

            return False

        # New candidate
        if (
            new_position
            != self.position_candidate
        ):

            self.position_candidate = (
                new_position
            )

            self.position_count = 1

            return False

        # Same candidate continues
        self.position_count += 1

        # Confirm after 5 frames
        if (
            self.position_count
            >= POSITION_CONFIRM_FRAMES
        ):

            old_position = self.position

            self.position = (
                self.position_candidate
            )

            self.position_candidate = None
            self.position_count = 0

            print(
                f"Track {self.track_id}: "
                f"{old_position} -> "
                f"{self.position}"
            )

            return True

        return False


# =========================
# TRACK ASSIGNMENT
# =========================

def assign_tracks(
    detections,
    tracks
):

    assignments = {}

    used_tracks = set()

    for detection_index, detection in enumerate(
        detections
    ):

        center = detection["center"]
        box = detection["box"]

        best_track = None
        best_distance = float("inf")

        for track_id, track in tracks.items():

            if track_id in used_tracks:

                continue

            distance = np.linalg.norm(
                np.array(center)
                - np.array(track.center)
            )

            iou = calculate_iou(
                box,
                track.box
            )

            if (
                distance <= MAX_MATCH_DISTANCE
                or iou > 0.15
            ):

                if distance < best_distance:

                    best_distance = distance
                    best_track = track_id

        if best_track is not None:

            assignments[
                detection_index
            ] = best_track

            used_tracks.add(
                best_track
            )

    return assignments


# =========================
# SPEECH
# =========================

def announce_track(track):

    if track.identity is None:

        return

    speech_manager.announce_person(
        track.identity,
        track.position
    )

    track.has_been_announced = True


def announce_position_change(track):

    if track.identity is None:

        return

    speech_manager.announce_person(
        track.identity,
        track.position
    )


# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    speech_manager.stop()

    raise SystemExit


print()
print("==============================")
print("SecondEye started")
print("==============================")
print("Q = Quit")
print("T = Read text")
print("==============================")
print()


# =========================
# TRACKING STATE
# =========================

tracks = {}

next_track_id = 0

frame_count = 0


# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "Could not read frame."
        )

        break

    frame_count += 1

    frame_height, frame_width = (
        frame.shape[:2]
    )

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

    # =========================
    # NO FACES
    # =========================

    if len(faces) == 0:

        for track in tracks.values():

            track.missed_frames += 1

        expired_tracks = [
            track_id
            for track_id, track
            in tracks.items()
            if (
                track.missed_frames
                > MAX_MISSED_FRAMES
            )
        ]

        for track_id in expired_tracks:

            del tracks[track_id]

        cv2.putText(
            frame,
            "No person detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    else:

        # =========================
        # DETECTIONS
        # =========================

        detections = []

        for (
            x,
            y,
            w,
            h
        ) in faces:

            center = (
                x + w // 2,
                y + h // 2
            )

            detections.append({
                "box": (x, y, w, h),
                "center": center
            })

        # =========================
        # ASSIGN TRACKS
        # =========================

        assignments = assign_tracks(
            detections,
            tracks
        )

        matched_track_ids = set()

        # =========================
        # UPDATE TRACKS
        # =========================

        for (
            detection_index,
            detection
        ) in enumerate(detections):

            center = detection["center"]
            box = detection["box"]

            if (
                detection_index
                in assignments
            ):

                track_id = assignments[
                    detection_index
                ]

                track = tracks[
                    track_id
                ]

                track.update(
                    center,
                    box
                )

                matched_track_ids.add(
                    track_id
                )

            else:

                track_id = next_track_id

                next_track_id += 1

                track = FaceTrack(
                    track_id,
                    center,
                    box
                )

                tracks[
                    track_id
                ] = track

                matched_track_ids.add(
                    track_id
                )

        # =========================
        # UNMATCHED TRACKS
        # =========================

        for track_id, track in tracks.items():

            if (
                track_id
                not in matched_track_ids
            ):

                track.missed_frames += 1

        # =========================
        # REMOVE EXPIRED
        # =========================

        expired_tracks = [
            track_id
            for track_id, track
            in tracks.items()
            if (
                track.missed_frames
                > MAX_MISSED_FRAMES
            )
        ]

        for track_id in expired_tracks:

            del tracks[track_id]

        # =========================
        # FACE RECOGNITION
        # =========================

        if (
            frame_count
            % RECOGNITION_INTERVAL
            == 0
        ):

            for track_id in matched_track_ids:

                if track_id not in tracks:

                    continue

                track = tracks[
                    track_id
                ]

                x, y, w, h = track.box

                face_crop = frame[
                    max(0, y):
                    min(
                        frame_height,
                        y + h
                    ),

                    max(0, x):
                    min(
                        frame_width,
                        x + w
                    )
                ]

                if face_crop.size == 0:

                    continue

                try:

                    result = (
                        DeepFace.represent(
                            img_path=face_crop,
                            model_name="Facenet512",
                            enforce_detection=False,
                            detector_backend="skip"
                        )
                    )

                    if not result:

                        continue

                    embedding = np.asarray(
                        result[0]["embedding"],
                        dtype=np.float32
                    )

                    best_identity = "Unknown"
                    best_distance = float(
                        "inf"
                    )

                    # -------------------------
                    # Compare with database
                    # -------------------------

                    for (
                        identity,
                        stored_embedding
                    ) in database_entries:

                        stored_embedding = (
                            np.asarray(
                                stored_embedding,
                                dtype=np.float32
                            )
                        )

                        # Ignore malformed embeddings
                        if (
                            embedding.shape
                            != stored_embedding.shape
                        ):

                            continue

                        distance = cosine_distance(
                            embedding,
                            stored_embedding
                        )

                        if (
                            distance
                            < best_distance
                        ):

                            best_distance = (
                                distance
                            )

                            best_identity = (
                                identity
                            )

                    # -------------------------
                    # Unknown threshold
                    # -------------------------

                    if (
                        best_distance
                        > THRESHOLD
                    ):

                        best_identity = (
                            "Unknown"
                        )

                    # -------------------------
                    # Stable identity
                    # -------------------------

                    (
                        identity_confirmed,
                        old_identity
                    ) = track.add_recognition(
                        best_identity,
                        best_distance
                    )

                    if identity_confirmed:

                        # First identity
                        if not track.has_been_announced:

                            announce_track(
                                track
                            )

                        # Identity changed
                        elif (
                            old_identity
                            != track.identity
                        ):

                            print(
                                "Identity changed:",
                                old_identity,
                                "->",
                                track.identity
                            )

                            speech_manager.announce_person(
                                track.identity,
                                track.position
                            )

                            track.has_been_announced = (
                                True
                            )

                except Exception as e:

                    print(
                        "Recognition error:",
                        e
                    )

        # =========================
        # POSITION TRACKING
        # =========================

        for track_id in matched_track_ids:

            if track_id not in tracks:

                continue

            track = tracks[
                track_id
            ]

            position_changed = (
                track.update_position(
                    frame_width
                )
            )

            if position_changed:

                if track.has_been_announced:

                    announce_position_change(
                        track
                    )

        # =========================
        # DRAW TRACKS
        # =========================

        for track_id, track in tracks.items():

            if track.missed_frames > 0:

                continue

            x, y, w, h = track.box

            # -------------------------
            # Box color
            # -------------------------

            if track.identity == "Unknown":

                box_color = (
                    0,
                    165,
                    255
                )

            elif track.identity is None:

                box_color = (
                    255,
                    255,
                    0
                )

            else:

                box_color = (
                    0,
                    255,
                    0
                )

            # -------------------------
            # Bounding box
            # -------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2
            )

            # -------------------------
            # Label
            # -------------------------

            if track.identity is None:

                identity_text = (
                    "Detecting..."
                )

            else:

                identity_text = (
                    track.identity
                )

            label = (
                f"{identity_text} | "
                f"{track.position}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x,
                    max(
                        25,
                        y - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2
            )

            # -------------------------
            # Track ID
            # -------------------------

            cv2.putText(
                frame,
                f"ID: {track_id}",
                (
                    x,
                    y + h + 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                1
            )

        # =========================
        # POSITION GUIDE
        # =========================

        left_line = int(
            frame_width
            * LEFT_ENTER
        )

        right_line = int(
            frame_width
            * RIGHT_ENTER
        )

        cv2.line(
            frame,
            (left_line, 0),
            (
                left_line,
                frame_height
            ),
            (255, 255, 255),
            1
        )

        cv2.line(
            frame,
            (right_line, 0),
            (
                right_line,
                frame_height
            ),
            (255, 255, 255),
            1
        )

        # -------------------------
        # Zone labels
        # -------------------------

        cv2.putText(
            frame,
            "LEFT",
            (20, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "CENTER",
            (
                frame_width // 2 - 40,
                frame_height - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "RIGHT",
            (
                frame_width - 80,
                frame_height - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # =========================
    # STATUS
    # =========================

    cv2.putText(
        frame,
        f"People: {len(tracks)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "T: Read Text",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    # =========================
    # DISPLAY
    # =========================

    cv2.imshow(
        "SecondEye",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # =========================
    # QUIT
    # =========================

    if key == ord("q"):

        break

    # =========================
    # OCR
    # =========================

    elif key == ord("t"):

        print(
            "Reading text..."
        )

        speech_manager.announce_text(
            "Reading text."
        )

        ocr_queue.put(
            frame.copy()
        )


# =========================
# SHUTDOWN
# =========================

print(
    "Shutting down..."
)

cap.release()

cv2.destroyAllWindows()

ocr_queue.put(None)

speech_manager.stop()

print(
    "SecondEye stopped."
)