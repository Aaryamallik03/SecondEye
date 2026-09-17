import pickle
import numpy as np

with open("data/database/embeddings.pkl", "rb") as f:
    database = pickle.load(f)

print("Database loaded.")
print("Type:", type(database))
print("Number of entries:", len(database))

for i, entry in enumerate(database):

    print("\nEntry", i + 1)
    print("Type:", type(entry))

    if isinstance(entry, dict):

        print("Keys:", entry.keys())

        for key, value in entry.items():

            if isinstance(value, (list, np.ndarray)):
                print(
                    key,
                    "type:", type(value),
                    "shape:", np.array(value).shape
                )
            else:
                print(key, "=", value)

    else:

        print("Shape:", np.array(entry).shape)