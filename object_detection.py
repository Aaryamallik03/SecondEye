import cv2
from ultralytics import YOLO


print("Loading YOLO...")

model = YOLO("yolo11s.pt")

print("Model loaded.")
print("Opening camera...")


cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Camera could not be opened.")
    raise SystemExit


while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read frame.")
        break


    results = model(
        frame,
        conf=0.50,
        imgsz=640,
        verbose=False
    )


    result = results[0]


    for box in result.boxes:

        confidence = float(box.conf[0])

        class_id = int(box.cls[0])

        label = model.names[class_id]


        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )


        text = (
            f"{label} "
            f"{confidence:.2f}"
        )


        cv2.putText(
            frame,
            text,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    cv2.imshow(
        "SecondEye - Detection Test",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


cap.release()

cv2.destroyAllWindows()

print("Test finished.")