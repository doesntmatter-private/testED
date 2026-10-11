"""Click the closed lane polygon and the crew direction on the first frame, save lane.json.

Usage:
  python -m laneguard.polygon_tool --source ../data/video/workzone_01.mp4 --out lane.json

Left click: add polygon vertex (4 or more).
Right click: set crew direction. Click once at the lane entry, once near the crew.
Enter: save.  r: reset.  q: quit without saving.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .capture import open_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", type=Path, default=Path("lane.json"))
    ap.add_argument("--frame", type=int, default=0, help="frame index to use as the setup snapshot")
    args = ap.parse_args()

    cap = open_source(args.source)
    frame = None
    for _ in range(args.frame + 1):
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("could not read frame")
    cap.release()

    poly: list[tuple[int, int]] = []
    direction: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            poly.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and len(direction) < 2:
            direction.append((x, y))

    win = "polygon_tool  L-click: lane vertex   R-click: entry then crew   Enter: save   r: reset   q: quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = frame.copy()
        if poly:
            cv2.polylines(vis, [np.array(poly, np.int32)], len(poly) > 2, (0, 140, 255), 2)
            for p in poly:
                cv2.circle(vis, p, 5, (0, 140, 255), -1)
        if len(direction) == 2:
            cv2.arrowedLine(vis, direction[0], direction[1], (0, 0, 255), 3, tipLength=0.08)
        elif len(direction) == 1:
            cv2.circle(vis, direction[0], 6, (0, 0, 255), -1)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            poly.clear(); direction.clear()
        if k in (13, 10):
            if len(poly) < 3 or len(direction) != 2:
                print("need at least 3 polygon points and a 2 point direction")
                continue
            (x0, y0), (x1, y1) = direction
            out = {
                "polygon": poly,
                "crew_direction": [x1 - x0, y1 - y0],
                "confirm_frames": 4,
                "heading_min_speed_px": 2.0,
                "cooldown_s": 8.0,
                "frame_size": [frame.shape[1], frame.shape[0]],
                "source": args.source,
            }
            args.out.write_text(json.dumps(out, indent=2))
            cv2.imwrite(str(args.out.with_suffix(".snapshot.jpg")), frame)
            print(f"saved {args.out} and setup snapshot")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
