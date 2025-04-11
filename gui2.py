import matplotlib.pyplot as plt
import torch
from collections import Counter
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np
import cv2
import io
import yt_dlp
import tempfile
import os

from model import EmotionRecognitionCNN  # Make sure emotionCNN.py is in the same directory

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

def plot_pie_chart(counter, exclude_neutral=False):
    '''Plot a pie chart of emotion distribution. Optionally exclude 'neutral'.'''
    fig = plt.figure(figsize=(6, 6), facecolor='#f9f9f9')
    items = counter.items()
    if exclude_neutral:
        items = [(k, v) for k, v in items if k != 'neutral']
    labels, values = zip(*items)
    wedges, texts, autotexts = plt.pie(
        values,
        labels=None,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        startangle=140,
        colors=plt.cm.Set3.colors,
        wedgeprops=dict(width=0.5, edgecolor='w')
    )
    
    plt.title("Emotion Distribution" + (" (No Neutral)" if exclude_neutral else ""), color='white')
    plt.axis('equal')
    legend = plt.legend(wedges, labels, title="Emotions", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), labelcolor='white', facecolor='#2c2c2c', edgecolor='none')
    legend.get_title().set_color('white')
    plt.gca().set_facecolor('#f9f9f9')
    plt.tight_layout()
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    fig.patch.set_facecolor('#f9f9f9')

def predict_by_video(video_path):
    '''Analyze a local video file frame-by-frame and return emotion distribution charts.'''
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    emotion_counts = []
    frame_index = 0
    analyzed = 0
    
    print(f"[DEBUG] Starting analysis on video: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % fps == 0:
             # print(f"[DEBUG] Analyzing frame {frame_index}/{total_frames}")
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
                    # print(f"[DEBUG] Detected emotion: {emotion}")
                    emotion_counts.append(emotion)
                break
            analyzed += 1
            # Removed progress bar from photo and video analysis to avoid unnecessary UI elements
        frame_index += 1
    cap.release()
    print(f"[DEBUG] Finished analyzing video. Total frames processed: {frame_index}, Faces analyzed: {analyzed}")

    counter = Counter(emotion_counts)
    emotions, counts = list(counter.keys()), list(counter.values())

     # Bar chart
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1f1f1f')
    bars = ax.bar(emotions, counts, color=plt.cm.Set3.colors[:len(emotions)])
    ax.set_facecolor('#1f1f1f')
    fig.patch.set_facecolor('#1f1f1f')
    ax.set_title("Emotion Counts", fontsize=14, weight='bold', color='white')
    ax.legend([bar.get_label() for bar in bars], emotions, loc='upper right', frameon=False, labelcolor='white')
    ax.tick_params(axis='x', labelsize=10, colors='white')
    ax.tick_params(axis='y', labelsize=10, colors='white')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    bar_image = Image.open(buf)
    plt.close()

    # Pie chart (all)
    pie_buf = io.BytesIO()
    plot_pie_chart(counter)
    plt.savefig(pie_buf, format='png', facecolor='#1f1f1f')
    pie_buf.seek(0)
    pie_all = Image.open(pie_buf)
    plt.close()

    # Pie chart (no neutral)
    pie_non_buf = io.BytesIO()
    plot_pie_chart(counter, exclude_neutral=True)
    plt.savefig(pie_non_buf, format='png', facecolor='#1f1f1f')
    pie_non_buf.seek(0)
    pie_non = Image.open(pie_non_buf)
    plt.close()

    return bar_image, pie_all, pie_non

def predict_emotion(image_np):
    '''Predict the emotion from an image with face detection.''' 
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return "No face detected."

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    img = transform(Image.fromarray(face)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        emotion = emotion_classes[predicted.item()]

    return emotion

def download_youtube_video(youtube_url, start_time=None, end_time=None):
    '''Download a YouTube video as low-res MP4 for fast processing using download_sections if available.'''
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "video.mp4")

    download_section = f"*{start_time}-{end_time}" if start_time and end_time else None

    ydl_opts = {
        'format': 'worst[ext=mp4][height<=480]',
        'outtmpl': output_path,
        'quiet': True,
        'download_sections': download_section if download_section else None,
        'force_keyframes_at_cuts': True,
        'verbose': True
    }
    print("[DEBUG] yt_dlp options:", ydl_opts)
    import time
    start_time_download = time.time()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print("[DEBUG] Downloading video with yt_dlp...")
        try:
            ydl.download([youtube_url])
        except Exception as e:
            print("[ERROR] yt_dlp download failed:", e)
        print("[DEBUG] Download complete. Output saved to:", output_path)
    print("[DEBUG] Download took {:.2f} seconds".format(time.time() - start_time_download))
    return output_path

def analyze_youtube_video(url, start_time=None, end_time=None, preview_html=None):
    '''Download and analyze a YouTube video by URL.'''
    video_path = download_youtube_video(url, start_time, end_time)
    iframe = generate_youtube_iframe(url)
    return (*predict_by_video(video_path), iframe)

def generate_youtube_iframe(url):
    '''Return iframe HTML for YouTube preview.'''
    video_id = url.split("v=")[-1].split("&")[0]
    return f"""<iframe width='100%' height='315' src='https://www.youtube.com/embed/{video_id}' \
    style='border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.1);' frameborder='0' allowfullscreen></iframe>"""


demo = gr.TabbedInterface(
    [
        gr.Interface(
        fn=predict_emotion,
        inputs=gr.Image(label="Upload Face Image", type="numpy"),
            outputs=gr.Text(label="Predicted Emotion"),
            title="Emotion Detection",
            description="Upload a 48x48 grayscale or color image of a face to detect the emotion."
        ),
        gr.Interface(
            fn=predict_by_video,
            inputs=gr.Video(label="Upload video"),
            outputs=[
            gr.Image(type="pil", label="Bar Chart"),
            gr.Image(type="pil", label="Pie Chart (All Emotions)"),
            gr.Image(type="pil", label="Pie Chart (No Neutral)"),
            gr.HTML(label="YouTube Preview"),           
            ]
        ),
        gr.Interface(
        fn=analyze_youtube_video,
        inputs=[
            gr.Text(label="YouTube Video URL"),
            gr.Text(label="Start Time (hh:mm:ss)", value="00:00:00"),
            gr.Text(label="End Time (hh:mm:ss)", value="00:00:10"),
            gr.HTML(label="Preview")
        ],
            outputs=[
                gr.Image(type="pil", label="Bar Chart"),
                gr.Image(type="pil", label="Pie Chart (All Emotions)"),
                gr.Image(type="pil", label="Pie Chart (No Neutral)")
            ]
        )
    ],
    ["Image Upload", "Video Upload", "YouTube Video"]
)

if __name__ == "__main__":
    demo.launch(share=True)
