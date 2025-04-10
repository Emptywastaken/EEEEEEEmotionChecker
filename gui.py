import matplotlib.pyplot as plt
import torch
from collections import Counter
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np
import cv2
import io

from model import EmotionRecognitionCNN  # Make sure model.py is in the same directory

# Emotion classes
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionRecognitionCNN().to(device)
model.load_state_dict(torch.load("emotion_model3_58.pth", map_location=device))
model.eval()

# Define preprocessing for image input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def plot_pie_chart(counter):
    plt.figure(figsize=(6, 6))
    plt.pie(
        list(counter.values()),
        labels=list(counter.keys()),
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Pastel1.colors
    )
    plt.title("Emotion Distribution (Pie Chart)")
    plt.axis('equal')  # ensures pie is round
    plt.tight_layout()


def predict_by_video(video_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) #frames per second
    emotion_counts = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % fps == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face_tensor = transform(Image.fromarray(face)).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(face_tensor)
                    _, predicted = torch.max(output, 1)
                    emotion = emotion_classes[predicted.item()]
                    emotion_counts.append(emotion)
                break  # predict only one face per frame for simplicity
        frame_index += 1
    cap.release()
    print(emotion_counts)
    counter = Counter(emotion_counts)
    emotions = list(counter.keys())
    counts = list(counter.values())

    # BAR CHART
    plt.figure(figsize=(10,5))
    plt.bar(emotions, counts, color='pink')
    plt.title("Emotion Frequency in Video")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()

    # PIE CHART
    pie_buf = io.BytesIO()
    plot_pie_chart(counter)
    plt.savefig(pie_buf, format='png')
    pie_buf.seek(0)
    pie_image = Image.open(pie_buf)
    plt.close()

    return img  # return buffer to Gradio


# Prediction function
def predict_emotion(image_np):

    # Convert NumPy array to PIL image
    img = Image.fromarray(image_np).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        emotion = emotion_classes[predicted.item()]

    return emotion

# Gradio UI
demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(label="Upload Face Image", type="numpy"),
    outputs=gr.Text(label="Predicted Emotion"),
    title="Emotion Detection",
    description="Upload a 48x48 grayscale or color image of a face to detect the emotion."
)

video_upload = gr.Interface(
    fn=predict_by_video,
    inputs=gr.Video(label="Upload video"),
    outputs=gr.Image(type="pil", label="Emotion Chart")
)

app = gr.TabbedInterface(
    [demo, video_upload],
    ["Image Upload", "Video Upload"]
)

if __name__ == "__main__":
    app.launch(share=True)

