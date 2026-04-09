import atexit
import cv2
import io
import os
import shutil
import time
import zipfile
import numpy as np
import gradio as gr
from PIL import Image
from sam3.visualization_utils import render_masklet_frame
from sam3_autolabeler import SAM3Autolabeler

TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


def cleanup_temp():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


atexit.register(cleanup_temp)

predictor = SAM3Autolabeler()

_state = {
    "overlays": [],           # pre-rendered overlay frames (list of np.ndarray)
    "masks": {},              # combined binary masks per frame {frame_idx: np.ndarray}
    "current_idx": 0,
    "num_frames": 0,
}


def extract_zip_to_tempdir(zip_path: str) -> str:
    """Extract a zip of JPEGs to a temp directory. Returns the directory path."""
    temp_dir = os.path.join(TEMP_DIR, f"frames_{time.time_ns()}")
    os.makedirs(temp_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)
    # Find JPEG files (may be in a subdirectory)
    jpeg_files = []
    for root, _, files in os.walk(temp_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg")):
                jpeg_files.append(os.path.join(root, f))
    if not jpeg_files:
        raise gr.Error("No JPEG files found in the zip.")
    # If all JPEGs are in a subdirectory, return that subdirectory
    parents = set(os.path.dirname(f) for f in jpeg_files)
    if len(parents) == 1:
        return parents.pop()
    return temp_dir


def load_frames_from_dir(frames_dir: str) -> list[np.ndarray]:
    """Load JPEG frames from a directory, sorted by filename."""
    jpeg_files = sorted([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg"))
    ])
    frames = []
    for path in jpeg_files:
        img = cv2.imread(path)
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def run_segmentation(zip_file, text_prompt: str):
    if zip_file is None:
        raise gr.Error("Please upload a zip file of JPEG frames.")
    if not text_prompt:
        raise gr.Error("Please enter a text prompt.")

    # Extract zip to temp directory
    frames_dir = extract_zip_to_tempdir(zip_file)

    # Load frames for visualization
    frames = load_frames_from_dir(frames_dir)
    if not frames:
        raise gr.Error("No valid JPEG frames found in the zip.")

    # Run SAM3 inference (SAM3 accepts a JPEG directory)
    predictor.load_video(video_path=frames_dir)
    outputs_per_frame = predictor.segment(text_prompt=text_prompt)

    # Pre-render all overlay frames and combine masks
    overlays = []
    masks = {}
    for i in range(len(frames)):
        if i in outputs_per_frame:
            out = outputs_per_frame[i]
            overlays.append(render_masklet_frame(frames[i], out, frame_idx=i))
            # Combine all object masks into one binary mask
            binary_masks = out["out_binary_masks"]
            if hasattr(binary_masks, "cpu"):
                binary_masks = binary_masks.cpu().numpy()
            if len(binary_masks) > 0:
                combined = np.any(binary_masks, axis=0).astype(np.uint8)
            else:
                combined = np.zeros(frames[i].shape[:2], dtype=np.uint8)
            if combined.ndim == 3:
                combined = combined.squeeze(0)
            masks[i] = combined
        else:
            overlays.append(frames[i])
            masks[i] = np.zeros(frames[i].shape[:2], dtype=np.uint8)

    _state["overlays"] = overlays
    _state["masks"] = masks
    _state["current_idx"] = 0
    _state["num_frames"] = len(frames)

    num_objects = len(outputs_per_frame.get(0, {}).get("out_obj_ids", []))
    status_msg = f"Done: {len(frames)} frames, {num_objects} object(s) detected"

    return overlays[0], f"Frame 0 / {len(frames) - 1}", status_msg


def step_forward():
    if _state["num_frames"] == 0:
        raise gr.Error("No segmentation results.")
    idx = min(_state["current_idx"] + 1, _state["num_frames"] - 1)
    _state["current_idx"] = idx
    return _state["overlays"][idx], f"Frame {idx} / {_state['num_frames'] - 1}"


def step_backward():
    if _state["num_frames"] == 0:
        raise gr.Error("No segmentation results.")
    idx = max(_state["current_idx"] - 1, 0)
    _state["current_idx"] = idx
    return _state["overlays"][idx], f"Frame {idx} / {_state['num_frames'] - 1}"


def play_video(fps):
    if _state["num_frames"] == 0:
        raise gr.Error("No segmentation results.")
    fps = max(1, int(fps))
    delay = 1.0 / fps
    for idx in range(_state["current_idx"], _state["num_frames"]):
        _state["current_idx"] = idx
        yield _state["overlays"][idx], f"Frame {idx} / {_state['num_frames'] - 1}"
        time.sleep(delay)


def download_masks():
    if not _state["masks"]:
        raise gr.Error("No segmentation results.")

    # Create a zip of NPZ files in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for frame_idx in sorted(_state["masks"].keys()):
            mask = _state["masks"][frame_idx]
            npz_buffer = io.BytesIO()
            np.savez_compressed(npz_buffer, mask=mask)
            npz_buffer.seek(0)
            zf.writestr(f"frame_{frame_idx:05d}.npz", npz_buffer.read())

    zip_path = os.path.join(TEMP_DIR, "masks.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_buffer.getvalue())

    return zip_path


with gr.Blocks() as demo:
    gr.Markdown("# SAM3 Autolabeler")

    with gr.Row():
        zip_input = gr.File(label="Upload zip of JPEG frames", file_types=[".zip"])
        text_prompt = gr.Textbox(label="Text prompt")

    run_button = gr.Button("Run segmentation")
    status = gr.Textbox(label="Status", interactive=False)

    frame_display = gr.Image(label="Frame with mask overlay", type="numpy")
    frame_label = gr.Textbox(label="Frame info", interactive=False)

    with gr.Row():
        back_button = gr.Button("< Prev")
        play_button = gr.Button("Play")
        forward_button = gr.Button("Next >")
        fps_input = gr.Number(label="Playback FPS", value=30, minimum=1, maximum=120)

    download_button = gr.Button("Download masks (.zip)")
    download_file = gr.File(label="Download", visible=False)

    run_button.click(
        fn=run_segmentation,
        inputs=[zip_input, text_prompt],
        outputs=[frame_display, frame_label, status],
    )

    forward_button.click(
        fn=step_forward,
        outputs=[frame_display, frame_label],
    )

    back_button.click(
        fn=step_backward,
        outputs=[frame_display, frame_label],
    )

    play_button.click(
        fn=play_video,
        inputs=fps_input,
        outputs=[frame_display, frame_label],
    )

    download_button.click(
        fn=download_masks,
        outputs=download_file,
    )

demo.launch()
