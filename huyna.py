import gradio as gr
import yt_dlp
import os
import tempfile
import torch
import cv2
import io
import matplotlib.pyplot as plt
from collections import Counter
from torchvision import transforms
from PIL import Image
import numpy as np

from model import EmotionRecognitionCNN

# Emotion labels
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load model
model = EmotionRecognitionCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("emotion_model3_58.pth", map_location=device))
model.to(device)
model.eval()

# Preprocess face image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def predict_emotion(image_np):
    '''Predict the emotion from a single face image (uploaded or from webcam).'''
    try:
        img = Image.fromarray(image_np).convert("RGB")
        img.save("debug_input.jpg")

        # Detect face using OpenCV cascade
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            print("No face detected in webcam image.")
            return "No face detected."

        x, y, w, h = faces[0]  # Only first face
        face = gray[y:y + h, x:x + w]
        face = cv2.equalizeHist(face)  # Enhance contrast
        face_img = Image.fromarray(face).resize((48, 48))
        face_img.save("debug_input_enhanced.jpg")

        img_tensor = transform(face_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            emotion = emotion_classes[predicted.item()]

            print("[DEBUG] Emotion prediction:")
            for idx, label in enumerate(emotion_classes):
                print(f"  {label:<9}: {probs[0, idx].item():.4f}")
            print(f"Top Prediction: {emotion} ({confidence.item():.2%})")

        return f"{emotion} ({confidence.item():.2%})"
    except Exception as e:
        print("Error during prediction:", e)
        return "Prediction failed"
    except Exception as e:
        print("Error during prediction:", e)
        return "Prediction failed"

def download_youtube_segment(url, start, end):
    '''Download a trimmed video segment from YouTube using yt_dlp.'''
    from datetime import datetime
    import time
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "video.mp4")

    section = f"*{start}-{end}"
    ydl_opts = {
        'outtmpl': output_path,
        'quiet': True,
        'download_sections': [section],
        'merge_output_format': 'mp4',
        'format': 'worst[ext=mp4][height<=480]',
        'retries': 3,
        'fragment_retries': 3,
        'socket_timeout': 30
    }
    try:
        print(f"[DEBUG] Downloading YouTube segment: {url} [{start} to {end}]")
        print("[DEBUG] Download options:", ydl_opts)
        print("[DEBUG] Download started at:", datetime.now().strftime('%H:%M:%S'))
        start_time = time.time()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        elapsed = time.time() - start_time
        print(f"[DEBUG] Download complete in {elapsed:.2f} seconds")
        print("[DEBUG] Downloaded file size:", os.path.getsize(output_path) / 1024 / 1024, "MB")
    except Exception as e:
        print("[ERROR] Failed to download video segment:", e)
        raise RuntimeError("Download failed")

    return output_path

def analyze_video(video_path, sample_every_seconds=5):
    '''Analyze a video file frame-by-frame and generate emotion distribution charts.'''
    import time
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    print(f"[DEBUG] Starting analysis on video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_rate = int(fps * sample_every_seconds) if fps > 0 else 30
    emotion_counts = []
    frame_index = 0
    start_time = time.time()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    analyzed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % skip_rate == 0:
            print(f"[DEBUG] Analyzing frame {frame_index} / {frame_count}")
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
                analyzed_frames += 1
                break
        frame_index += 1
    cap.release()
    elapsed = time.time() - start_time
    print(f"[DEBUG] Video analysis complete in {elapsed:.2f} seconds")
    print(f"[DEBUG] Total frames analyzed: {analyzed_frames} / {frame_count}")
    elapsed = time.time() - start_time
    print(f"[DEBUG] Video analysis complete in {elapsed:.2f} seconds")

    counter = Counter(emotion_counts)
    emotions = list(counter.keys())
    counts = list(counter.values())

    plt.style.use('ggplot')
    plt.figure(figsize=(10, 4))
    plt.bar(emotions, counts)
    plt.title("Detected Emotions - Bar Chart")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    bar_chart = Image.open(buf)
    plt.close()

    # Pie chart (all emotions)
    pie_buf = io.BytesIO()
    plt.figure(figsize=(5, 5))
    plt.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title("Emotion Distribution (All Emotions)")
    plt.tight_layout()
    plt.savefig(pie_buf, format='png')
    pie_buf.seek(0)
    pie_all = Image.open(pie_buf)
    plt.close()

    # Pie chart (exclude neutral)
    no_neutral = [(emo, cnt) for emo, cnt in zip(emotions, counts) if emo != 'neutral']
    if no_neutral:
        pie_no_neutral_buf = io.BytesIO()
        labels, values = zip(*no_neutral)
        plt.figure(figsize=(5, 5))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title("Emotion Distribution (No Neutral)")
        plt.tight_layout()
        plt.savefig(pie_no_neutral_buf, format='png')
        pie_no_neutral_buf.seek(0)
        pie_non = Image.open(pie_no_neutral_buf)
        plt.close()
    else:
        pie_non = pie_all
    plt.title("Detected Emotions")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return bar_chart, pie_all, pie_non

def show_youtube_preview(url):
    '''Generate YouTube iframe preview from a video URL.'''
    video_id = url.split("v=")[-1].split("&")[0]
    iframe = f"""
    <iframe width='100%' height='315' src='https://www.youtube.com/embed/{video_id}' 
    frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture' 
    allowfullscreen></iframe>
    """
    return iframe

def full_pipeline(url, start, end, sample_every, video_file_path):
    '''Run the full pipeline: trim YouTube video or use uploaded file, then analyze emotions.'''
    if video_file_path is not None:
        print("[DEBUG] Using uploaded video file")
        video_path = video_file_path
    else:
        video_path = download_youtube_segment(url, start, end)
    return analyze_video(video_path, sample_every)

with gr.Blocks() as demo:
    gr.Markdown("### Emotion Detection from Image or YouTube Clip")

    with gr.TabItem("Image (Upload or Webcam)"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="Upload or Capture Image")
                image_button = gr.Button("Detect Emotion")
            progress = gr.Textbox(label="Progress", interactive=False, visible=False)
        emotion_output = gr.Text(label="Predicted Emotion")
        # Removed duplicate bloc
        def predict_emotion_with_progress(image_np):
    '''Wrapper for emotion prediction with future progress feedback.'''
    return predict_emotion(image_np)

        image_button.click(fn=predict_emotion_with_progress, inputs=image_input, outputs=emotion_output)

    with gr.TabItem("YouTube Trim & Analyze"):
        gr.Markdown("Paste a YouTube link OR upload a local video file")
        youtube_url = gr.Text(label="YouTube Link", placeholder="https://www.youtube.com/watch?v=...")
        video_file = gr.File(label="Or Upload Video File (.mp4)", file_types=[".mp4"])
        preview_html = gr.HTML()
        preview_btn = gr.Button("Show YouTube Preview")
        preview_btn.click(fn=show_youtube_preview, inputs=youtube_url, outputs=preview_html)

        with gr.Row():
            start_time = gr.Text(label="Start (hh:mm:ss)", value="00:00:00")
            end_time = gr.Text(label="End (hh:mm:ss)", value="00:00:10")

        frame_rate_slider = gr.Slider(minimum=1, maximum=30, step=1, value=5, label="Sample Every N Seconds")
        analyze_btn = gr.Button("Download & Analyze Clip")
        output_bar = gr.Image(label="Bar Chart", visible=False)
        output_pie_all = gr.Image(label="Pie Chart (All Emotions)", visible=False)
        output_pie_non = gr.Image(label="Pie Chart (No Neutral)", visible=False)
        analyze_btn.click(fn=full_pipeline, inputs=[youtube_url, start_time, end_time, frame_rate_slider, video_file], outputs=[output_bar, output_pie_all, output_pie_non])

if __name__ == "__main__":
    demo.launch(share=True)
