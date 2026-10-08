import cv2
from deepface import DeepFace

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not access camera")
    exit()

print("Camera opened.")
print("Press S to analyze one frame.")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        break

    cv2.imshow("Test Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        print("\nAnalyzing face...")

        try:
            result = DeepFace.represent(
                img_path=frame,
                model_name="Facenet512",
                detector_backend="opencv",
                enforce_detection=True
            )

            print("Face embedding created successfully!")
            print("Embedding length:", len(result[0]["embedding"]))

        except Exception as e:
            print("Recognition error:")
            print(e)

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()