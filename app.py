from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify, session
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import threading
import face_recognition
import pickle
import cv2
import numpy as np
import time
import secrets
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.secret_key = secrets.token_hex(16)  # Generate a random secret key for sessions

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Path to your face encodings file
ENCODINGS_PATH = "face_encodings_4.pickle"

# Dictionary to track processing status for each session
session_status = {}


def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def face_distance_to_conf(face_distance):
    """Convert a face distance to a confidence percentage"""
    if face_distance > 0.6:
        return 0
    # Scale the match percentage - the closer to 0, the better the match
    # 0 distance = 100% match, 0.6 distance = 0% match
    return (1 - (face_distance / 0.6)) * 100


def resize_image_if_needed(image, max_height=1000):
    """Resize image if it's too large (speeds up processing)"""
    height, width = image.shape[:2]
    print(f'{height} x {width}')
    if height > max_height:
        # Calculate the ratio of the height and construct the dimensions
        ratio = max_height / float(height)
        new_width = int(width * ratio)
        # Resize the image
        return cv2.resize(image, (new_width, max_height))
    return image


def process_image_thread(image_path, original_filename, session_id):
    """Process the image in a separate thread"""
    global session_status

    try:
        session_status[session_id]['is_processing'] = True
        session_status[session_id]['current_image'] = original_filename
        session_status[session_id]['progress'] = 5
        session_status[session_id]['message'] = 'Starting image processing...'
        session_status[session_id]['face_matches'] = []  # Reset face matches

        # Check if encodings file exists
        if not os.path.isfile(ENCODINGS_PATH):
            session_status[session_id]['message'] = 'Face encodings file not found!'
            session_status[session_id]['is_processing'] = False
            return

        # Load the uploaded image
        session_status[session_id]['progress'] = 10
        session_status[session_id]['message'] = 'Loading image...'
        image = cv2.imread(image_path)
        if image is None:
            session_status[session_id]['message'] = 'Failed to load image'
            session_status[session_id]['is_processing'] = False
            return

        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image to speed up processing
        session_status[session_id]['progress'] = 20
        session_status[session_id]['message'] = 'Optimizing image size...'
        rgb_small = resize_image_if_needed(rgb_image)

        # Detect faces
        session_status[session_id]['progress'] = 30
        session_status[session_id]['message'] = 'Detecting faces...'
        face_locations = face_recognition.face_locations(rgb_small)

        if not face_locations:
            session_status[session_id]['message'] = 'No faces found in the image'
            session_status[session_id]['is_processing'] = False
            return

        # Get face encodings
        session_status[session_id]['progress'] = 50
        session_status[session_id]['message'] = f'Found {len(face_locations)} faces. Encoding...'
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        # Load known face encodings
        session_status[session_id]['progress'] = 60
        session_status[session_id]['message'] = 'Loading known faces...'
        with open(ENCODINGS_PATH, "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)

        # Match faces
        session_status[session_id]['progress'] = 70
        session_status[session_id]['message'] = 'Matching faces with database...'

        # Calculate scaling ratio if image was resized
        h_ratio = rgb_image.shape[0] / rgb_small.shape[0]
        w_ratio = rgb_image.shape[1] / rgb_small.shape[1]

        matches_list = []
        all_face_matches = []  # Store all face matches data

        # Process each detected face
        for i, face_encoding in enumerate(face_encodings):
            session_status[session_id]['message'] = f'Matching face {i + 1} of {len(face_encodings)}...'

            # Calculate face distance to all known faces
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            # Convert distances to confidence scores
            matches = []
            for j, face_distance in enumerate(face_distances):
                confidence = face_distance_to_conf(face_distance)
                matches.append((known_face_names[j], confidence))

            # Sort matches by confidence (highest first)
            matches.sort(key=lambda x: x[1], reverse=True)

            # Store top 5 matches for display on the web page
            top_matches = matches[:5]
            all_face_matches.append({
                'face_index': i + 1,
                'matches': [(name, f"{conf:.2f}") for name, conf in top_matches]
            })

            print(f"\nTop matches for face {i + 1}:")
            for match in top_matches:
                print(f"- {match[0]} (Confidence: {match[1]:.2f}%)")

            # Select best match if confidence > 0
            if matches and matches[0][1] > 0:
                matches_list.append(matches[0])
            else:
                matches_list.append(("Unknown", 0))

        # Store all face matches in the processing status
        session_status[session_id]['face_matches'] = all_face_matches

        # Create output image with recognized faces
        session_status[session_id]['progress'] = 85
        session_status[session_id]['message'] = 'Creating result image...'

        # Draw rectangles and labels on the original image
        for (top, right, bottom, left), (name, confidence) in zip(face_locations, matches_list):
            # Scale back to original image size if resized
            top = int(top * h_ratio)
            right = int(right * w_ratio)
            bottom = int(bottom * h_ratio)
            left = int(left * w_ratio)

            # Draw rectangle around face
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 4)

            # Add text with name and confidence
            text = f"{name} ({confidence:.2f}%)"
            cv2.putText(image, text, (left, bottom + 90),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)

        # Save the output image
        session_status[session_id]['progress'] = 95
        session_status[session_id]['message'] = 'Saving result...'

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"{session_id}_{timestamp}.jpg"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)

        cv2.imwrite(result_path, image)

        session_status[session_id]['progress'] = 100
        session_status[session_id]['message'] = 'Processing complete'
        session_status[session_id]['result_file'] = result_filename

    except Exception as e:
        session_status[session_id]['message'] = f"Error: {str(e)}"

    finally:
        session_status[session_id]['is_processing'] = False


@app.route('/')
def index():
    """Render the main page"""
    # Create a new session_id if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

    session_id = session['session_id']

    # Initialize session status if not exists
    if session_id not in session_status:
        session_status[session_id] = {
            'is_processing': False,
            'current_image': None,
            'progress': 0,
            'message': '',
            'result_file': None,
            'face_matches': [],
            'latest_upload': None
        }

    # Get status for this session
    status = session_status[session_id]

    return render_template('index.html',
                           latest_upload=status.get('latest_upload'),
                           latest_result=status.get('result_file'),
                           is_processing=status.get('is_processing', False),
                           processing_progress=status.get('progress', 0),
                           processing_message=status.get('message', ''),
                           face_matches=status.get('face_matches', []),
                           session_id=session_id)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle the file upload and process the image"""
    # Get or create session_id
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    session_id = session['session_id']

    # Initialize session status if not exists
    if session_id not in session_status:
        session_status[session_id] = {
            'is_processing': False,
            'current_image': None,
            'progress': 0,
            'message': '',
            'result_file': None,
            'face_matches': [],
            'latest_upload': None
        }

    # If already processing, don't start another job
    if session_status[session_id]['is_processing']:
        return redirect(url_for('index', error="Already processing an image. Please wait."))

    if 'file' not in request.files:
        return redirect(url_for('index', error="No file part"))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index', error="No selected file"))

    if file and allowed_file(file.filename):
        # Add session_id and timestamp to filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = f"{session_id}_{timestamp}{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the uploaded file
        file.save(filepath)

        # Store the upload filename in session status
        session_status[session_id]['latest_upload'] = filename

        # Process the image in a separate thread
        processing_thread = threading.Thread(
            target=process_image_thread,
            args=(filepath, file.filename, session_id)
        )
        processing_thread.daemon = True
        processing_thread.start()

        return redirect(url_for('index'))

    return redirect(url_for('index', error="Invalid file type"))


@app.route('/status')
def processing_status_endpoint():
    """Return the current processing status as JSON"""
    if 'session_id' not in session:
        return jsonify({
            'is_processing': False,
            'progress': 0,
            'message': 'No active session'
        })

    session_id = session['session_id']
    if session_id not in session_status:
        return jsonify({
            'is_processing': False,
            'progress': 0,
            'message': 'No status for this session'
        })

    return jsonify(session_status[session_id])


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve the uploaded files"""
    # Security check: Make sure the file belongs to the current session
    if 'session_id' in session and filename.startswith(session['session_id']):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    return "Unauthorized", 403


@app.route('/results/<filename>')
def result_file(filename):
    """Serve the result files"""
    # Security check: Make sure the file belongs to the current session
    if 'session_id' in session and filename.startswith(session['session_id']):
        return send_from_directory(app.config['RESULTS_FOLDER'], filename)
    return "Unauthorized", 403


@app.route('/clear-session')
def clear_session():
    """Clear the current session"""
    if 'session_id' in session:
        session_id = session['session_id']
        if session_id in session_status:
            del session_status[session_id]
        session.pop('session_id', None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    # Create the HTML template
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition Upload</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2 {
            color: #333;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .upload-form {
            background-color: white;
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .results-container, .matches-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
            background-color: white;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-comparison {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }
        .image-box {
            flex: 1;
            min-width: 300px;
            text-align: center;
            border: 1px solid #eee;
            padding: 10px;
            border-radius: 5px;
            background-color: #fafafa;
        }
        .image-box img {
            max-width: 100%;
            height: auto;
            border-radius: 3px;
        }
        .no-image {
            padding: 50px;
            background-color: #f0f0f0;
            color: #888;
            border-radius: 3px;
        }
        input[type="file"] {
            margin: 10px 0;
            width: 100%;
        }
        input[type="submit"], button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        input[type="submit"]:hover, button:hover {
            background-color: #45a049;
        }
        .error {
            color: red;
            background-color: #ffeeee;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .processing-container {
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            text-align: center;
        }
        .progress-bar {
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            margin: 10px 0;
            overflow: hidden;
        }
        .progress-bar-fill {
            height: 100%;
            background-color: #4CAF50;
            width: 0%;
            transition: width 0.5s;
            border-radius: 10px;
        }
        h3 {
            margin-top: 5px;
            margin-bottom: 5px;
        }
        @media (max-width: 600px) {
            .image-comparison {
                flex-direction: column;
            }
            .image-box {
                min-width: unset;
            }
        }
        .hidden {
            display: none;
        }
        #refresh-btn {
            background-color: #2196F3;
            color: white;
            padding: 5px 10px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
        }
        .match-list {
            list-style-type: none;
            padding: 0;
            margin: 0;
        }
        .match-item {
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        .match-item:last-child {
            border-bottom: none;
        }
        .face-matches {
            margin-top: 15px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        .face-title {
            background-color: #e1f5fe;
            padding: 8px;
            border-radius: 5px;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .new-session-btn {
            background-color: #ff9800;
            margin-top: 20px;
        }
    </style>
    <script>
        // Function to check processing status periodically
        function checkStatus() {
            if (!document.getElementById('processing-container').classList.contains('hidden')) {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        // Update progress bar
                        document.getElementById('progress-bar-fill').style.width = data.progress + '%';

                        // Update status message
                        document.getElementById('processing-message').textContent = data.message;

                        // If processing is complete, refresh the page
                        if (data.progress >= 100 || !data.is_processing) {
                            setTimeout(() => {
                                window.location.reload();
                            }, 1000);
                        } else {
                            // Check again in 1 second
                            setTimeout(checkStatus, 1000);
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching status:', error);
                        setTimeout(checkStatus, 3000); // Try again in 3 seconds
                    });
            }
        }

        // Initialize when the page loads
        window.onload = function() {
            // If processing is happening, start the status check
            if (!document.getElementById('processing-container').classList.contains('hidden')) {
                checkStatus();
            }
        };

        // Function to clear session and start new
        function startNewSession() {
            window.location.href = '/clear-session';
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>Face Recognition</h1>

        {% if error %}
        <div class="error">
            {{ error }}
        </div>
        {% endif %}

        {% if request.args.get('error') %}
        <div class="error">
            {{ request.args.get('error') }}
        </div>
        {% endif %}

        <button onclick="startNewSession()" class="new-session-btn">Start New Session</button>

        <div class="upload-form" {% if is_processing %}style="opacity: 0.5;"{% endif %}>
            <h2>Upload a new image</h2>
            <form action="/upload" method="post" enctype="multipart/form-data" {% if is_processing %}onsubmit="return false;"{% endif %}>
                <input type="file" name="file" accept="image/*" capture="camera" {% if is_processing %}disabled{% endif %}>
                <br>
                <input type="submit" value="Upload and Recognize Faces" {% if is_processing %}disabled{% endif %}>
            </form>
        </div>

        <div id="processing-container" class="processing-container {% if not is_processing %}hidden{% endif %}">
            <h2>Processing Image</h2>
            <div class="progress-bar">
                <div id="progress-bar-fill" class="progress-bar-fill" style="width: {{ processing_progress }}%;"></div>
            </div>
            <p id="processing-message">{{ processing_message }}</p>
            <button id="refresh-btn" onclick="window.location.reload();">Refresh Page</button>
        </div>

        <div class="results-container">
            <h2>Latest Results</h2>

            <div class="image-comparison">
                <div class="image-box">
                    <h3>Original Image</h3>
                    {% if latest_upload %}
                    <img src="{{ url_for('uploaded_file', filename=latest_upload) }}" alt="Original Image">
                    {% else %}
                    <div class="no-image">No image uploaded yet</div>
                    {% endif %}
                </div>

                <div class="image-box">
                    <h3>Recognized Faces</h3>
                    {% if latest_result %}
                    <img src="{{ url_for('result_file', filename=latest_result) }}" alt="Face Recognition Result">
                    {% else %}
                    <div class="no-image">No processed image yet</div>
                    {% endif %}
                </div>
            </div>
        </div>

        {% if face_matches %}
        <div class="matches-container">
            <h2>Face Match Details</h2>

            {% for face in face_matches %}
            <div class="face-matches">
                <div class="face-title">Top matches for face {{ face.face_index }}:</div>
                <ul class="match-list">
                    {% for name, confidence in face.matches %}
                    <li class="match-item">- {{ name }} (Confidence: {{ confidence }}%)</li>
                    {% endfor %}
                </ul>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>
</body>
</html>
        ''')

    print("Starting face recognition web server with session support...")
    print(f"Make sure the '{ENCODINGS_PATH}' file exists in the current directory")

    # Run the app on all network interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)