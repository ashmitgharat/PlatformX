import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
import warnings
warnings.filterwarnings('ignore')  # Suppress deprecation warnings
import cv2
import numpy as np
import sys
# Redirect stderr during DeepFace import
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from deepface import DeepFace
sys.stderr.close()
sys.stderr = original_stderr
from fpdf import FPDF
import matplotlib.pyplot as plt
import speech_recognition as sr
from textblob import TextBlob

def analyze_face_video(video_or_frame, audio_path, output_pdf, real_time=False):
    if real_time:
        frame = video_or_frame
        try:
            result = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False, silent=True)[0]
            return {
                'age': result['age'],
                'gender': result['dominant_gender'],
                'gender_confidence': result['gender'][result['dominant_gender']],
                'emotion': result['dominant_emotion'],
                'emotion_confidence': result['emotion'][result['dominant_emotion']]
            }
        except Exception as e:
            return {'error': str(e)}

    # Full video analysis
    cap = cv2.VideoCapture(video_or_frame)
    face_data = {"male": 0, "female": 0, "avg_age": 0, "emotions": {}, "frames": 0}
    audio_data = analyze_audio(audio_path) if audio_path else {"transcript": "", "sentiment": 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False, silent=True)[0]
            gender = result['dominant_gender']
            if gender.lower() == "male":
                face_data["male"] += 1
            else:
                face_data["female"] += 1
            face_data["avg_age"] += result['age']
            emotion = result['dominant_emotion']
            face_data["emotions"][emotion] = face_data["emotions"].get(emotion, 0) + 1
            face_data["frames"] += 1
        except:
            continue

    cap.release()
    face_data["avg_age"] = face_data["avg_age"] / face_data["frames"] if face_data["frames"] > 0 else 0
    generate_pdf(face_data, audio_data, output_pdf)
    return face_data

def analyze_audio(audio_input):
    recognizer = sr.Recognizer()
    if isinstance(audio_input, str):  # File path
        with sr.AudioFile(audio_input) as source:
            audio = recognizer.record(source)
    else:  # Audio data
        audio = audio_input
    try:
        text = recognizer.recognize_google(audio)
        sentiment = TextBlob(text).sentiment.polarity
        return text, sentiment
    except sr.UnknownValueError:
        return "Could not understand audio", 0

def generate_pdf(face_data, audio_data, output_pdf):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "AI Interview Companion Report", ln=True, align='C')
    pdf.cell(200, 10, f"Gender: Male: {face_data['male']}, Female: {face_data['female']}", ln=True)
    pdf.cell(200, 10, f"Average Age: {face_data['avg_age']:.1f}", ln=True)
    pdf.cell(200, 10, "Emotion Distribution:", ln=True)

    plt.figure(figsize=(5, 3))
    plt.bar(face_data["emotions"].keys(), face_data["emotions"].values(), color='blue')
    plt.title("Emotion Analysis")
    chart_path = "static/reports/emotion_chart.png"
    plt.savefig(chart_path)
    pdf.image(chart_path, x=50, w=100)

    pdf.cell(200, 10, f"Transcript: {audio_data['transcript']}", ln=True)
    pdf.cell(200, 10, f"Sentiment Score: {audio_data['sentiment']:.2f}", ln=True)
    pdf.output(output_pdf)