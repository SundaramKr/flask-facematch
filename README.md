# Face Recognition System

A high-performance face recognition system built with Python, Flask, OpenCV, and dlib. This system supports real-time multi-face detection against a pre-encoded Pickle database, featuring a web interface for uploading and analyzing high-resolution images.

## Features

- **Real-Time Webcam Recognition**: Live face tracking and matching using webcam feed, optimized with 1-in-9 frame sampling at 0.25x resolution for smooth FPS (`livematch.py`).
- **Web Upload Interface**: A Flask web application capable of handling high-resolution image uploads (up to 16 MB).
- **Background Processing**: Implements threaded background image processing with AJAX progress polling, ensuring the server remains unblocked during heavy encoding tasks.
- **Session-Based Access Control**: Flask session management allows concurrent multi-user image analysis securely.
- **Pre-encoded Database**: Compares detected faces against a pre-encoded Pickle database of known faces for rapid matching.

## Prerequisites

- Python 3.8+
- C++ Build Tools (required for building `dlib` on Windows)
- CMake (required for `dlib`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SundaramKr/flask-facematch.git
   cd flaskserver
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Installing `dlib` may take some time as it compiles C++ code.*

## Usage

### 1. Web Application (Upload & Analyze)
Start the Flask web server to use the browser interface:
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`. You can upload images to be processed and matched against the database.

### 2. Live Webcam Recognition
To run the real-time face recognition via your webcam:
```bash
python livematch.py
```
Press `q` to quit the webcam stream.

### 3. Generate New Face Encodings
If you want to add new faces to the database:
1. Place images of known faces in a folder (filename will be used as the person's name).
2. Run the encoding script:
```bash
python enc.py
```
3. Follow the prompt to enter the folder path. It will generate a new `.pickle` database.

### 4. Single Image / Multiple Face Scripts
You can also run standalone scripts for specific tasks:
- `python matchfaces.py` - Match a single face from an image or webcam capture.
- `python multiplefaces.py` - Detect and match multiple faces in a single frame.

## Project Structure

- `app.py`: Main Flask application handling routing, threaded processing, and the web interface.
- `enc.py`: Utility script to generate face encodings from a directory of images.
- `livematch.py`: Script for high-FPS, real-time webcam face recognition.
- `matchfaces.py` / `multiplefaces.py`: Standalone scripts for testing recognition logic.
- `templates/`: Contains the HTML interface (`index.html`) with AJAX polling logic.

## Notice

This repository contains a lightweight sample database (`face_encodings_4.pickle`) for testing purposes. User-uploaded images (`uploads/`) and processed results (`results/`) are intentionally ignored by git to protect privacy.
