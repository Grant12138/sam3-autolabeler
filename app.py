import cv2
import os
import tempfile
import gradio as gr
from sam3.visualization_utils import save_masklet_video
from sam3_autolabeler import SAM3Autolabeler

predictor = SAM3Autolabeler()

def run_video_segmentation(video_path: str, text_prompt: str):
    if not os.path.isfile(video_path):
        raise gr.Error(f"File not found: {video_path}")
    if not video_path.endswith(".mp4"):
        raise gr.Error(f"Expected an mp4 file, but got {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    video_frames_for_vis = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    predictor.load_video(video_path=video_path)
    outputs_per_frame = predictor.segment(text_prompt=text_prompt)
    output_video_path = tempfile.mktemp(suffix=".mp4")
    save_masklet_video(video_frames_for_vis, outputs_per_frame, output_video_path)

    return output_video_path

with gr.Blocks() as demo:
    video_path = gr.Textbox(label="Enter the path to an mp4 file below")
    text_prompt = gr.Textbox(label="Enter a text prompt below")
    run_button = gr.Button("Run segmentation")
    output_video = gr.Video(label="Segmented video")

    run_button.click(
        fn=run_video_segmentation,
        inputs=[video_path, text_prompt],
        outputs=output_video
    )

demo.launch()
