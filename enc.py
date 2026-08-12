import face_recognition
import os
import pickle
from pathlib import Path


def create_encodings(faces_folder):
    known_face_encodings = []
    known_face_names = []

    print(f"Processing faces in {faces_folder}...")
    total_faces = len([f for f in os.listdir(faces_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    processed = 0

    for filename in os.listdir(faces_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(faces_folder, filename)

            try:
                # Load image
                image = face_recognition.load_image_file(image_path)

                # Get encodings
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    known_face_encodings.append(encodings[0])
                    name = Path(filename).stem  # Get filename without extension
                    known_face_names.append(name)
                else:
                    print(f"No face found in {filename}")

                # Progress update
                processed += 1
                if processed % 100 == 0 or processed == total_faces:
                    print(f"Processed {processed}/{total_faces} images")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return known_face_encodings, known_face_names


def main():
    # Prompt user for input
    faces_folder = input("Enter the path to the folder containing face images: ")

    if not os.path.isdir(faces_folder):
        print(f"Error: The folder {faces_folder} does not exist.")
        return

    # Generate encodings
    known_face_encodings, known_face_names = create_encodings(faces_folder)

    # Save the encodings and names to pickle file
    pickle_file = "face_encodings_4.pickle"
    with open(pickle_file, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    print(f"\nSuccess! {len(known_face_names)} face encodings saved to {pickle_file}")


if __name__ == "__main__":
    main()