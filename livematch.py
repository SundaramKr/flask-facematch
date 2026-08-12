import face_recognition
import os
import pickle
import cv2
import numpy as np
import time


def face_distance_to_conf(face_distance):
    """
    Convert a face distance to a confidence percentage
    """
    if face_distance > 0.6:
        return 0

    # Scale the match percentage - the closer to 0, the better the match
    # 0 distance = 100% match, 0.6 distance = 0% match
    return (1 - (face_distance / 0.6)) * 100


def capture_image_from_webcam(encodings_path):
    """
    Continuously capture frames from the webcam and display the matching name with the highest confidence live.
    """
    # Initialize the webcam (index 0 for the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    # Optimize camera settings - lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting live recognition...")

    # Load pre-computed encodings
    print(f"Loading encodings from {encodings_path}...")
    with open(encodings_path, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)

    # Variables for performance measurement and optimization
    last_frame_time = time.time()
    fps_counter = 0
    fps = 0
    frame_count = 0
    skip_frames = 9  # Process 1 frame every N frames for performance

    # Store previous results for frames we skip processing
    previous_results = []

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Calculate FPS
        fps_counter += 1
        current_time = time.time()
        if (current_time - last_frame_time) > 1.0:
            fps = fps_counter
            fps_counter = 0
            last_frame_time = current_time

        # Only process frames periodically to improve performance
        frame_count = (frame_count + 1) % skip_frames
        process_frame = (frame_count == 0)

        # Display results from previous processing for skipped frames
        if not process_frame and previous_results:
            # Reuse the previous results
            for (top, right, bottom, left), name, confidence in previous_results:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, f"{name} ({confidence:.1f}%)",
                            (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

        # Process frame for face recognition
        if process_frame:
            previous_results = []

            # Resize frame to speed up face detection (quarter size)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all face locations in the frame using HOG (faster)
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

            if face_locations:
                # Find face encodings
                try:
                    # Use the face_recognition API directly on the small frame
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    # Process each detected face
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Scale face locations to the original frame size
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        # Compare with known faces
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        confidence = face_distance_to_conf(face_distances[best_match_index])
                        name = known_face_names[best_match_index]

                        # Display recognition results
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, f"{name} ({confidence:.1f}%)",
                                    (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

                        # Store results for future frames
                        previous_results.append(((top, right, bottom, left), name, confidence))

                except Exception as e:
                    print(f"Error in face recognition: {e}")

        # Display FPS counter
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Break the loop when 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    encodings_path = "face_encodings_4.pickle"

    # Validate paths
    if not os.path.isfile(encodings_path):
        print(f"Error: The encodings file {encodings_path} does not exist.")
        return

    # Start live webcam recognition
    capture_image_from_webcam(encodings_path)


if __name__ == "__main__":
    main()