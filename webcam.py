import tkinter as tk  # not used here but left for compatibility
import cv2
import torch
import numpy as np
import time
from PIL import Image
from torchvision import transforms
from model import EmotionRecognitionCNN

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionRecognitionCNN().to(device)
model.load_state_dict(torch.load("emotion_model3_58.pth", map_location=device))
model.eval()

emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

stream = cv2.VideoCapture(0)
if not stream.isOpened():
    print("No stream")
    exit()

last_prediction_time = 0  # time of last prediction
emotion = ""  # store latest emotion

while True:
    ret, frame = stream.read()
    if not ret:
        print("No more stream.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    current_time = time.time()

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]

        if current_time - last_prediction_time >= 1:  # only predict every 1 second
            face_pil = Image.fromarray(face)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face_tensor)
                _, pred = torch.max(output, 1)
                emotion = emotion_classes[pred.item()]
                last_prediction_time = current_time

        # Always show latest emotion until next prediction
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()
