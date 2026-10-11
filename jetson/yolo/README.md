# laneguard (Option B)

YOLO11n + ByteTrack + polygon intrusion logic. Runs on a T400, 4060 Ti, or Jetson Orin Nano with the same code.

```
laneguard/
  app.py           main loop
  capture.py       file / RTSP / GStreamer source, auto hardware decode on Jetson
  intrusion.py     IntrusionState: containment, heading, confirmation, cooldown
  geometry.py      point in polygon, vector helpers (no deps)
  alerts.py        Jetson.GPIO relay, logs when GPIO is absent
  evidence.py      rolling buffer, writes 10 s clip + JSON per alarm
  polygon_tool.py  click the lane and crew direction, writes lane.json (desktop, needs a display)
  web.py           operator web interface served from the running process
  runtime.py       shared state between the processing loop and the web server
  static/          the single page web UI
  lcs.py           Caltrans Lane Closure System client, site metadata, event stamping
  diag.py          per track diagnostic against a lane file
  enhance.py       optional gamma + CLAHE low light preprocessing
tests/             unit tests for the pure Python logic, run anywhere
```

## Operator web interface

```bash
python -m laneguard.app --source rtsp://cam/stream --lane lanes/site.json \
  --model ../data/models/yolo11n_fp16.engine --web 8080
```

Open `http://<host>:8080` from any phone, tablet or laptop on the same network.

- **Live view** with detections, track ids, anchors and the active lane polygon, as an MJPEG stream.
- **Edit lane** freezes a raw frame, lets you tap the lane corners and the crew direction, and adjust confirm frames, min speed and cooldown. Save writes `lane.json` plus a setup snapshot and hot reloads the pipeline. No restart.
- **Events** lists alarms and near misses with a click to play the evidence clip.
- **Test alarm** fires the relay and records a clip so the crew can confirm the hardware at the start of a shift.
- **Caltrans closure** links the deployment to a record in the Caltrans Lane Closure System. Pick the district, search by route, place, work type or closure ID, or use Near me to list active closures within 5 miles, nearest first. The chosen record plus crew, operator and notes is saved as `<lane>.site.json` and stamped into every event JSON under `lcs`, so each intrusion ties to an official closure ID, route, postmile and type of work. Needs internet access to cwwp2.dot.ca.gov; the feed is cached for 5 minutes.
- `--loop` restarts a file source when it ends, useful for demos.

The web server runs in a thread inside the same process. It adds about 1 ms per frame for JPEG encoding.

## Quick start (T400 or any CUDA desktop, Linux)

```bash
cd yolo
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m unittest discover -s tests -v

# export engine once per GPU
yolo export model=yolo11n.pt format=engine half=True imgsz=640 device=0
mv yolo11n.engine ../data/models/yolo11n_t400_fp16.engine

# define the lane on your clip, then run
python -m laneguard.polygon_tool --source ../data/video/workzone_01.mp4 --out lane.json
python -m laneguard.app --source ../data/video/workzone_01.mp4 --lane lane.json \
  --model ../data/models/yolo11n_t400_fp16.engine --show
```

## T400 notes

- 2 GB variant: one stream only. Keep `--imgsz 640`, batch is already 1. Watch `nvidia-smi` on first run.
- No NVENC: evidence clips are encoded on the CPU with OpenCV. Fine for 10 s clips.
- Expect 30 to 50 fps inference, which is close to Orin Nano, so confirmation and heading tuning here roughly carries over.

## Jetson

Same commands, but create the venv with `--system-site-packages` so the GStreamer-enabled apt OpenCV is used, and install torch from `https://pypi.jetson-ai-lab.dev/jp6/cu126`. `capture.py` detects Jetson and switches to `nvv4l2decoder` automatically. See docs/dev-setup.md section 10.
