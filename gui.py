import gradio as gr
from predict import predict_emotion_with_progress
from video_utils import download_youtube_segment, analyze_video

def show_youtube_preview(url):
    '''Return an embeddable iframe HTML from a YouTube link.'''
    try:
        video_id = url.split("v=")[-1].split("&")[0]
        return f"""
        <iframe width='100%' height='315' src='https://www.youtube.com/embed/{video_id}'
        frameborder='0' allow='autoplay; encrypted-media' allowfullscreen></iframe>
        """
    except Exception as e:
        return f"<p style='color:red;'>Invalid YouTube URL: {str(e)}</p>"

with gr.Blocks() as demo:
    gr.Markdown("### Emotion Detection from Image or YouTube Clip")

    with gr.TabItem("Image (Upload or Webcam)"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="Upload or Capture Image")
                image_button = gr.Button("Detect Emotion")
            progress = gr.Textbox(label="Progress", interactive=False, visible=True)
        emotion_output = gr.Text(label="Predicted Emotion")

        image_button.click(
            fn=predict_emotion_with_progress,
            inputs=[image_input, progress],
            outputs=[emotion_output]
        )

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

        def full_pipeline(url, start, end, sample_every, video_file_path, progress_callback=None):
            '''Run full processing: trim video or use uploaded, analyze emotions.'''
            if video_file_path is not None:
                print("[DEBUG] Using uploaded video file")
                video_path = video_file_path
            else:
                video_path = download_youtube_segment(url, start, end)
            if progress_callback:
                progress_callback("Analyzing video...")
            result = analyze_video(video_path, sample_every)
            if progress_callback:
                progress_callback("Analysis complete.")
            return result

        analyze_btn.click(
            fn=full_pipeline,
            inputs=[youtube_url, start_time, end_time, frame_rate_slider, video_file, progress],
            outputs=[output_bar, output_pie_all, output_pie_non]
        )

if __name__ == "__main__":
    demo.launch(share=True)
