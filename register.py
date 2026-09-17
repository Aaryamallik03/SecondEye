import cv2
import os

name = input("Enter person's name: ").strip()

folder = os.path.join("data", "faces", name)
os.makedirs(folder, exist_ok=True)

camera = cv2.VideoCapture(0)

count = 0

print("Look at the camera. Capturing 10 images...")
print("Press Q to cancel.")

while count < 10:
    ret, frame = camera.read()

    if not ret:
        print("Could not access camera")
        break

    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        filename = os.path.join(folder, f"{count + 1}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
        print(f"Captured image {count}/10")

camera.release()
cv2.destroyAllWindows()

print(f"Registration complete for {name}")