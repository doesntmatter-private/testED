"""Diagnostic: run detector + tracker on a clip and summarise every track against a lane file.

Usage:
  python -m laneguard.diag --source clip.mp4 --lane lanes/x.json --model model.engine [--save-overlay out.jpg]

Prints per track: frames seen, first/last anchor, mean speed along crew direction, frames inside polygon.
Optionally saves one frame with the polygon and all anchor trajectories drawn, for checking the polygon placement.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .app import VEHICLE_CLASSES, load_lane
from .capture import open_source
from .enhance import enhance
from .geometry import dot, point_in_polygon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--lane", required=True, type=Path)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--enhance", action="store_true", help="gamma + CLAHE low light preprocessing")
    ap.add_argument("--save-overlay", type=Path, default=None)
    ap.add_argument("--overlay-frame", type=int, default=0)
    args = ap.parse_args()

    lane = load_lane(args.lane)
    model = YOLO(args.model)
    cap = open_source(args.source)
    tracks: dict[int, list[tuple[int, float, float, int]]] = defaultdict(list)  # (frame, x, y, cls)
    det_per_frame = []
    overlay = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n == args.overlay_frame:
            overlay = frame.copy()
        infer = enhance(frame) if args.enhance else frame
        res = model.track(infer, persist=True, tracker="bytetrack.yaml", classes=VEHICLE_CLASSES,
                          imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        k = 0
        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.cpu().numpy().astype(int)
            cls = res.boxes.cls.cpu().numpy().astype(int)
            for box, tid, c in zip(res.boxes.xyxy.cpu().numpy(), ids, cls):
                x1, y1, x2, y2 = box
                tracks[int(tid)].append((n, float((x1 + x2) / 2), float(y2), int(c)))
                k += 1
        det_per_frame.append(k)
        n += 1
    cap.release()

    print(f"frames={n} frame_size={lane and args.lane} detections/frame mean={np.mean(det_per_frame):.2f} max={max(det_per_frame) if det_per_frame else 0}")
    print(f"polygon={lane.polygon} crew_dir=({lane.crew_direction[0]:.2f},{lane.crew_direction[1]:.2f}) min_speed={lane.heading_min_speed_px}")
    print(f"{'id':>4} {'cls':>3} {'frames':>6} {'first_anchor':>16} {'last_anchor':>16} {'speed_along':>11} {'inside':>6} {'inside_inbound_run':>18}")
    for tid, pts in sorted(tracks.items()):
        f0, x0, y0, c = pts[0]
        f1, x1, y1, _ = pts[-1]
        nf = len(pts)
        speed = dot((x1 - x0, y1 - y0), lane.crew_direction) / max(1, f1 - f0)
        inside = sum(point_in_polygon((x, y), lane.polygon) for _, x, y, _ in pts)
        # longest run of consecutive inside frames with inbound heading over a 5 frame window
        best = run = 0
        for i in range(len(pts)):
            _, x, y, _ = pts[i]
            j = max(0, i - 5)
            _, xa, ya, _ = pts[j]
            sp = dot((x - xa, y - ya), lane.crew_direction) / max(1, i - j)
            if point_in_polygon((x, y), lane.polygon) and sp >= lane.heading_min_speed_px:
                run += 1; best = max(best, run)
            else:
                run = 0
        print(f"{tid:>4} {c:>3} {nf:>6} {f'({x0:.0f},{y0:.0f})':>16} {f'({x1:.0f},{y1:.0f})':>16} {speed:>11.1f} {inside:>6} {best:>18}")

    if args.save_overlay and overlay is not None:
        cv2.polylines(overlay, [np.array(lane.polygon, np.int32)], True, (0, 140, 255), 3)
        rng = np.random.default_rng(0)
        for tid, pts in tracks.items():
            col = tuple(int(v) for v in rng.integers(80, 255, 3))
            for i in range(1, len(pts)):
                cv2.line(overlay, (int(pts[i-1][1]), int(pts[i-1][2])), (int(pts[i][1]), int(pts[i][2])), col, 2)
            cv2.putText(overlay, str(tid), (int(pts[-1][1]), int(pts[-1][2]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_overlay), overlay)
        print(f"overlay saved to {args.save_overlay}")


if __name__ == "__main__":
    main()
