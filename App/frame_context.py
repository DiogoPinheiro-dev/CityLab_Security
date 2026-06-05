from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class FrameContext:
    original_frame: np.ndarray
    processing_frame: np.ndarray
    scale_x: float
    scale_y: float
    process_scale: float

    def map_bbox_to_original(self, bbox: list[int] | tuple[int, int, int, int]) -> list[int]:
        x1, y1, x2, y2 = bbox
        return [
            int(round(x1 * self.scale_x)),
            int(round(y1 * self.scale_y)),
            int(round(x2 * self.scale_x)),
            int(round(y2 * self.scale_y)),
        ]

    def map_point_to_original(self, point: list[float] | tuple[float, float]) -> list[int]:
        x, y = point
        return [
            int(round(x * self.scale_x)),
            int(round(y * self.scale_y)),
        ]

    def clip_original_bbox(self, bbox: list[int] | tuple[int, int, int, int]) -> list[int]:
        frame_h, frame_w = self.original_frame.shape[:2]
        x1, y1, x2, y2 = bbox
        return [
            max(0, min(frame_w, int(x1))),
            max(0, min(frame_h, int(y1))),
            max(0, min(frame_w, int(x2))),
            max(0, min(frame_h, int(y2))),
        ]


def build_frame_context(
    frame: np.ndarray,
    process_scale: float,
    experimental_grayscale: bool = False,
) -> FrameContext:
    frame_h, frame_w = frame.shape[:2]
    process_w = max(1, int(round(frame_w * process_scale)))
    process_h = max(1, int(round(frame_h * process_scale)))

    processing_frame = cv2.resize(frame, (process_w, process_h))
    if experimental_grayscale:
        gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
        processing_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return FrameContext(
        original_frame=frame,
        processing_frame=processing_frame,
        scale_x=frame_w / float(process_w),
        scale_y=frame_h / float(process_h),
        process_scale=process_scale,
    )
