import torch
import numpy as np
from sam3.model_builder import build_sam3_video_predictor

class SAM3Autolabeler:
    def __init__(self) -> None:
        """Build the SAM3 video predictor"""
        self._predictor = build_sam3_video_predictor(version="sam3.1")
        self._session_id = None

    def load_video(self, video_path: str) -> None:
        """Start an inference session on a video. Close any existing session first."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self._session_id is not None:
                self._predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=self._session_id,
                    )
                )
            response = self._predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=video_path,
                )
            )
            self._session_id = response["session_id"]

    def segment(self, text_prompt: str) -> dict[int, np.ndarray]:
        """Reset session, if applicable, apply text prompt on frame 0, and propogate through video.
        Return {frame_index: binary_mask_array} for all frames."""
        if self._session_id is None:
            raise RuntimeError("No video loaded. Call load_video() first.")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._predictor.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=self._session_id,
                )
            )

            frame_idx = 0
            self._predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=self._session_id,
                    frame_index=frame_idx,
                    text=text_prompt,
                )
            )

            outputs_per_frame = {}
            for response in self._predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=self._session_id,
                )
            ):
                outputs_per_frame[response["frame_index"]] = response["outputs"]

            return outputs_per_frame

    def close(self) -> None:
        """Close the current session and shutdown the predictor"""
        if self._session_id is not None:
            self._predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=self._session_id,
                )
            )
        self._predictor.shutdown()
