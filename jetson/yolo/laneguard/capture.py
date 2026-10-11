"""Frame source abstraction: file, RTSP, or GStreamer string on Jetson."""
from __future__ import annotations

import platform

import cv2


def is_jetson() -> bool:
    try:
        return "tegra" in platform.release() or open("/proc/device-tree/model").read().lower().startswith("nvidia")
    except OSError:
        return False


def open_source(src: str, use_gstreamer: bool | None = None) -> cv2.VideoCapture:
    """Open a video file path, an RTSP URL, or a raw GStreamer pipeline.

    On Jetson, hardware decode via nvv4l2decoder is used automatically.
    On a desktop GPU (T400, 4060 Ti) OpenCV from pip decodes on the CPU, which is fine for one stream.
    """
    if use_gstreamer is None:
        use_gstreamer = is_jetson()

    if src.startswith("gst:"):
        return cv2.VideoCapture(src[4:], cv2.CAP_GSTREAMER)

    if use_gstreamer:
        if src.startswith("rtsp://"):
            pipe = (f"rtspsrc location={src} latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder "
                    "! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR "
                    "! appsink drop=true max-buffers=1")
        else:
            pipe = (f"filesrc location={src} ! qtdemux ! h264parse ! nvv4l2decoder "
                    "! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR "
                    "! appsink")
        return cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)

    cap = cv2.VideoCapture(src)
    if src.startswith("rtsp://"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
