import cv2
import torch
import numpy as np
import os
import pandas as pd
from torchvision import transforms
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from fpdf import FPDF
from datetime import timedelta
from emotionCNN import EmotionCNN  # replace with actual filename

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("emotion_modelv2WORKING.pth", map_location=device))
model.eval()

# Emotion labels
label_map = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ----- Transform -----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----- Load Face Detector -----
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----- Analyze Video -----
video_path = "speed.mp4"
cap = cv2.VideoCapture(video_path)
emotion_predictions = []
frame_interval = 30  # sample every 30 frames (~1s at 30fps)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print(f"No faces detected at frame {frame_count}.")  # Debugging: No faces detected

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_pil = transforms.functional.to_pil_image(face)
            input_tensor = transform(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, 1).item()
                emotion_predictions.append(label_map[pred])
            break  # one face is enough

    frame_count += 1

cap.release()

# Debugging: Check if emotion_predictions is empty
if len(emotion_predictions) == 0:
    print("No emotions were predicted. Please check face detection and model inference.")
else:
    print(f"Emotion predictions collected for {len(emotion_predictions)} frames.")

# ----- Create Charts -----
os.makedirs('report_images', exist_ok=True)

# Pie Chart
if emotion_predictions:
    emotion_counts = Counter(emotion_predictions)
    emotions, counts = zip(*emotion_counts.items())
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=140, colors=plt.cm.Set3.colors)
    plt.title('Emotion Distribution')
    plt.tight_layout()
    plt.savefig("report_images/emotion_pie.png")
    plt.close()

    # Timeline Chart
    df = pd.DataFrame({'emotion': emotion_predictions})
    df['time_sec'] = df.index
    df['emotion_code'] = df['emotion'].astype('category').cat.codes
    plt.figure(figsize=(10, 4))
    plt.plot(df['time_sec'], df['emotion_code'], marker='o', linestyle='-', color='royalblue')
    plt.yticks(df['emotion_code'].unique(), df['emotion'].astype('category').cat.categories)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Emotion')
    plt.title('Emotion Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("report_images/emotion_timeline.png")
    plt.close()

    # ----- Generate PDF Report -----
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Emotion Analysis Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Total Frames Analyzed: {len(emotion_predictions)}", ln=True)
    pdf.cell(200, 10, txt=f"Dominant Emotion: {emotion_counts.most_common(1)[0][0]}", ln=True)
    pdf.ln(10)
    pdf.image("report_images/emotion_pie.png", x=30, w=150)

    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Emotion Timeline", ln=True, align='C')
    pdf.image("report_images/emotion_timeline.png", x=15, w=180)

    pdf.output("emotion_report.pdf")
    print("✅ PDF report saved as 'emotion_report.pdf'")
else:
    print("No emotions were predicted, skipping chart and PDF generation.")
