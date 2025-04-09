import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from emotionCNN import EmotionCNN  
import time
# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("emotion_modelv2WORKING.pth", map_location=device))
model.eval()


emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# FAce detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define transform
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Start video capture
cap = cv2.VideoCapture(0)

last_prediction_time = time.time()
current_emotion = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_pil = Image.fromarray(face)

        # Only predict once per second
        if time.time() - last_prediction_time >= 1:
            input_tensor = transform(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                current_emotion = emotions[predicted.item()]
                last_prediction_time = time.time()

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, current_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()