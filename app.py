import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
import warnings
warnings.filterwarnings('ignore')  # Suppress deprecation warnings
import cv2
import time
import threading
import configparser
import sounddevice as sd
import numpy as np
import wave
from flask import Flask, render_template, request, send_file, jsonify, Response
from flask_socketio import SocketIO
import speech_recognition as sr
from face_analyzer import analyze_face_video, analyze_audio

# Redirect stderr during import to suppress remaining TensorFlow warnings
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
app = Flask(__name__)
socketio = SocketIO(app)
sys.stderr.close()
sys.stderr = original_stderr

# Load config
config = configparser.ConfigParser()
config.read('config.ini')
FRAME_SKIP = config.getint('settings', 'frame_skip')
PORT = config.getint('settings', 'port')

# Global Variables
recording = False
timestamp = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_VIDEO = config['paths']['output_video'].format(timestamp=timestamp)
OUTPUT_AUDIO = config['paths']['output_audio'].format(timestamp=timestamp)
REPORT_PATH = config['paths']['report_path'].format(timestamp=timestamp)

# Ensure directories exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/reports", exist_ok=True)

# Real-time face detection and analysis
def face_detection():
    global recording
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 10.0, (640, 480))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    frame_count = 0

    while recording:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1
        if frame_count % FRAME_SKIP == 0:
            try:
                result = analyze_face_video(frame, None, None, real_time=True)
                socketio.emit('live_update', result)
            except Exception as e:
                socketio.emit('live_update', {'error': str(e)})

        cv2.imshow("Interview Companion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Audio recording using sounddevice
def record_audio():
    global recording
    SAMPLE_RATE = 44100  # Standard sample rate for speech recognition
    CHANNELS = 1  # Mono audio
    DURATION = 1  # Record in 1-second chunks

    audio_data = []
    def callback(indata, frames, time_info, status):
        if recording:
            audio_data.append(indata.copy())
        return (0, pyaudio.paContinue) if recording else (0, pyaudio.paComplete)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
        while recording:
            time.sleep(DURATION)  # Record in chunks

    # Save as WAV file
    if audio_data:
        audio_array = np.concatenate(audio_data, axis=0)
        with wave.open(OUTPUT_AUDIO, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit (2 bytes)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_array.tobytes())

    # Analyze audio in real-time for live updates
    if os.path.exists(OUTPUT_AUDIO):
        with sr.AudioFile(OUTPUT_AUDIO) as source:
            audio = sr.Recognizer().record(source)
            text, sentiment = analyze_audio(audio)
            socketio.emit('audio_update', {'text': text, 'sentiment': sentiment})

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_recording():
    global recording
    if recording:
        return jsonify({"error": "Already recording"}), 400
    recording = True
    threading.Thread(target=face_detection).start()
    threading.Thread(target=record_audio).start()
    return jsonify({"message": "Recording started"})

@app.route('/stop', methods=['POST'])
def stop_recording():
    global recording
    if not recording:
        return jsonify({"error": "Not recording"}), 400
    recording = False
    time.sleep(2)
    try:
        analyze_face_video(OUTPUT_VIDEO, OUTPUT_AUDIO, REPORT_PATH)
        return jsonify({"message": "Recording stopped. Report generated.", "report_available": True})
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}", "report_available": False}), 500

@app.route('/download')
def download_report():
    if os.path.exists(REPORT_PATH):
        response = send_file(REPORT_PATH, as_attachment=True)
        # os.remove(REPORT_PATH)  # Uncomment to clean up after download
        return response
    return jsonify({"error": "Report not found"}), 404

# Video feed for live preview
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, debug=True, port=PORT, allow_unsafe_werkzeug=True)