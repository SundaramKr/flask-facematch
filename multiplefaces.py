import face_recognition
import os
import pickle
import cv2

def face_distance_to_conf(face_distance):
    """
    Convert a face distance to a confidence percentage
    """
    if face_distance > 0.6:
        return 0

    # Scale the match percentage - the closer to 0, the better the match
    # 0 distance = 100% match, 0.6 distance = 0% match
    return (1 - (face_distance / 0.6)) * 100

def capture_image_from_webcam():
    """
    Capture an image from the webcam and return the image file path.
    """
    # Initialize the webcam (index 0 for the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Press 'c' to capture the image")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the captured frame
        cv2.imshow("Webcam", frame)

        # Wait for user to press 'c' to capture the image
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save the captured image to a file
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image captured and saved as {image_path}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return image_path


def match_face(unknown_image_path, encodings_path):
    unknown_image = face_recognition.load_image_file(unknown_image_path)
    unknown_face_encodings = face_recognition.face_encodings(unknown_image)

    '''
    # Resize if larger than 800x800
    max_dim = 800
    height, width = unknown_image.shape[:2]
    if width > max_dim or height > max_dim:
        scaling_factor = min(max_dim / width, max_dim / height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        unknown_image = cv2.resize(unknown_image, (new_width, new_height))
        print(f"Resized image resolution: {new_width}x{new_height}")
    '''

    if not unknown_face_encodings:
        print(f"No faces found in the unknown image: {unknown_image_path}")
        return [], unknown_image

    print(f"Loading encodings from {encodings_path}...")
    with open(encodings_path, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)

    print(f"Loaded {len(known_face_names)} face encodings")

    matches_list = []

    for idx, unknown_face_encoding in enumerate(unknown_face_encodings):
        face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)

        matches = []
        for i, face_distance in enumerate(face_distances):
            confidence = face_distance_to_conf(face_distance)
            matches.append((known_face_names[i], confidence))

        # Sort matches highest confidence first
        matches.sort(key=lambda x: x[1], reverse=True)

        # Print top 5 matches for this face
        print(f"\nTop matches for face {idx + 1}:")
        for match in matches[:5]:
            print(f"- {match[0]} (Confidence: {match[1]:.2f}%)")

        # Select best match if confidence > 0
        if matches and matches[0][1] > 0:
            matches_list.append(matches[0])
        else:
            # No good match
            matches_list.append(("Unknown", 0))

    return matches_list, unknown_image

def show_result(image, matches_list):
    face_locations = face_recognition.face_locations(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for (top, right, bottom, left), (name, confidence) in zip(face_locations, matches_list):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        text = f"{name} ({confidence:.2f}%)"
        cv2.putText(image, text, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # Prompt user for input
    choice = int(input("Enter 1 to capture picture from webcam or 2 to use an image file: "))
    if choice == 1:
        unknown_image_path = capture_image_from_webcam()
        if not unknown_image_path:
            return
    elif choice == 2:
        unknown_image_path = input("Enter the path to your unknown face image: ")
        if not os.path.isfile(unknown_image_path):
            print(f"Error: The file {unknown_image_path} does not exist.")
            return
    else:
        print("Invalid Choice")
        return

    encodings_path = "face_encodings_4.pickle"

    if not os.path.isfile(encodings_path):
        print(f"Error: The encodings file {encodings_path} does not exist.")
        return

    print(f"Processing image {unknown_image_path}...")
    matching_names, unknown_image = match_face(unknown_image_path, encodings_path)

    if matching_names:
        print("\nMatches found:")
        for name, confidence in matching_names:
            print(f"- {name} (Confidence: {confidence:.2f}%)")

        # Show the image with overlay
        show_result(unknown_image, matching_names)
    else:
        print("No matches found")

if __name__ == "__main__":
    main()