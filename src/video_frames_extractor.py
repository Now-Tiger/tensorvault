#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import cv2


def extract_video_frames(video_path: str, output_dir: str, frame_rate: int = 1) -> list:
    """
    Extracts frames from a video file at a set interval (default: 1 frame per second).
    Returns a list of dictionaries containing the extracted file paths and matching timestamps.
    """
    _ = os.makedirs(output_dir, exist_ok=True)
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        video.release()
        raise ValueError("Could not read video properties or video has 0 FPS.")

    frame_interval = int(fps * frame_rate)

    extracted_frames = []
    frame_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            timestamp = round(frame_count / fps, 2)
            frame_filename = f"frame_{timestamp}s.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            # Save frame snapshot locally
            _ = cv2.imwrite(frame_path, frame)

            extracted_frames.append({"frame_path": frame_path, "timestamp": timestamp})

        frame_count += 1

    video.release()
    return extracted_frames
