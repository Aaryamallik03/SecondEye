from deepface import DeepFace
import os
import pickle

faces_folder = "data/faces"
database_folder = "data/database"

os.makedirs(database_folder, exist_ok=True)

embeddings = []

for person_name in os.listdir(faces_folder):

    person_folder = os.path.join(faces_folder, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"\nProcessing {person_name}...")

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet512",
                detector_backend="opencv",
                enforce_detection=True
            )

            embedding = result[0]["embedding"]

            embeddings.append({
                "name": person_name,
                "embedding": embedding
            })

            print(f"Processed: {image_name}")

        except Exception as e:
            print(f"Skipped {image_name}: {e}")

with open("data/database/embeddings.pkl", "wb") as file:
    pickle.dump(embeddings, file)

print("\nEmbeddings created successfully!")
print(f"Total embeddings: {len(embeddings)}")