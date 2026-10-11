"""Main loop: capture -> YOLO track -> intrusion logic -> alerts + evidence.

Usage:
  python -m laneguard.app --source ../data/video/workzone_01.mp4 --lane lane.json --show
  python -m laneguard.app --source rtsp://user:pass@cam/stream --lane lane.json --model yolo11n.engine
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from .alerts import Alerter
from .capture import open_source
from .enhance import enhance
from .evidence import Evidence
from .intrusion import Event, IntrusionState, LaneConfig
from .lcs import event_stamp, load_site
from .runtime import Runtime

VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
log = logging.getLogger("laneguard")


def load_lane(path: Path) -> LaneConfig:
    d = json.loads(Path(path).read_text())
    return LaneConfig(
        polygon=[tuple(p) for p in d["polygon"]],
        crew_direction=tuple(d["crew_direction"]),
        confirm_frames=d.get("confirm_frames", 4),
        heading_min_speed_px=d.get("heading_min_speed_px", 2.0),
        cooldown_s=d.get("cooldown_s", 8.0),
    )


def draw(frame, lane: LaneConfig, results, state: IntrusionState, fps: float):
    import numpy as np
    pts = np.array(lane.polygon, dtype=np.int32)
    thick = max(3, frame.shape[1] // 320)          # 4 px at 1280 wide, 6 px at 1920, 12 px at 4K
    cv2.polylines(frame, [pts], True, (0, 0, 255), thick, lineType=cv2.LINE_AA)
    if results.boxes is not None and results.boxes.id is not None:
        for box, tid in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.id.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
            cv2.circle(frame, ((x1 + x2) // 2, y2), max(4, thick), (0, 230, 255), -1)
            cv2.putText(frame, str(tid), (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{fps:5.1f} fps", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file, rtsp:// URL, or gst:<pipeline>")
    ap.add_argument("--lane", required=True, type=Path, help="lane.json from polygon_tool")
    ap.add_argument("--model", default="yolo11n.pt", help=".pt, .onnx or .engine")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--enhance", action="store_true", help="gamma + CLAHE low light preprocessing")
    ap.add_argument("--evidence", type=Path, default=Path("../data/events"))
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-alerts", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (testing)")
    ap.add_argument("--web", type=int, default=0, metavar="PORT", help="serve the operator web interface on this port")
    ap.add_argument("--loop", action="store_true", help="restart a file source when it ends (demo mode)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    lane = load_lane(args.lane)
    model = YOLO(args.model)
    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    state = IntrusionState(lane)
    alerter = None if args.no_alerts else Alerter()
    evidence = Evidence(args.evidence, fps=src_fps)

    rt = None
    if args.web:
        from .web import start_in_thread
        rt = Runtime(args.lane, args.evidence, args.source, args.model)
        start_in_thread(rt, port=args.web)

    n = 0
    t_start = time.time()
    t_begin = t_start
    fps_est = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.loop and not args.source.startswith(("rtsp://", "gst:")):
                    cap.release()
                    cap = open_source(args.source)
                    continue
                log.info("end of stream")
                break
            n += 1
            t = time.time()

            if rt is not None:
                reload_lane, test_alarm = rt.take_flags()
                if reload_lane:
                    try:
                        lane = load_lane(args.lane)
                        state = IntrusionState(lane)
                        log.info("lane reloaded: %d points", len(lane.polygon))
                    except Exception as e:  # keep running on a bad file
                        log.error("lane reload failed: %s", e)
                if test_alarm:
                    ev = Event("test", -1, t, (0.0, 0.0), 0, 0.0)
                    if alerter:
                        alerter.fire(ev)
                    rt.publish_event(ev, evidence.start_clip(ev, {"source": args.source, "test": True, **event_stamp(load_site(args.lane))}))

            infer = enhance(frame) if args.enhance else frame
            t_inf = time.time()
            res = model.track(infer, persist=True, tracker="bytetrack.yaml", classes=VEHICLE_CLASSES,
                              imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            infer_ms = (time.time() - t_inf) * 1000
            if res.boxes is not None and res.boxes.id is not None:
                for box, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy().astype(int)):
                    state.update(int(tid), tuple(map(float, box)), t)
            state.end_frame(t)
            evidence.push(frame)

            for ev in state.pop_events():
                clip = None
                if ev.kind == "alarm":
                    if alerter:
                        alerter.fire(ev)
                    clip = evidence.start_clip(ev, {"source": args.source, "model": args.model, **event_stamp(load_site(args.lane))})
                else:
                    log.info("near miss track=%d frames_inside=%d", ev.track_id, ev.frames_inside)
                if rt is not None:
                    rt.publish_event(ev, clip)

            if n % 30 == 0:
                fps_est = 30 / max(1e-6, time.time() - t_start)
                t_start = time.time()
            if rt is not None:
                annotated = draw(frame.copy(), lane, res, state, fps_est)
                rt.publish_frame(annotated, raw=frame)
                rt.publish_stats(n, fps_est, infer_ms, len(state.track_ids()))
            if args.show:
                cv2.imshow("laneguard", draw(frame, lane, res, state, fps_est))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        evidence.flush()
        cap.release()
        if alerter:
            alerter.close()
        cv2.destroyAllWindows()
        elapsed = max(1e-6, time.time() - t_begin)
        log.info("processed %d frames in %.1fs (%.1f fps end to end)", n, elapsed, n / elapsed)


if __name__ == "__main__":
    main()
