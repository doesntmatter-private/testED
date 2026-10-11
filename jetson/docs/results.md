# Test results log

Running record of pipeline runs. Newest at the bottom of each section.

## Environment: T400 4GB development box

| Item | Value |
|------|-------|
| Host | 10.111.72.128, Ubuntu 22.04.4, Python 3.10.12 |
| GPU | NVIDIA T400 4GB, driver 580.126.09 |
| Stack | torch 2.6.0+cu124, torchvision 0.21.0+cu124, ultralytics 8.4.175, OpenCV 5.0.0 |
| Engine | yolo11n, FP16, 640 px, TensorRT, 68 MB, 7 min build |
| Engine inference | 12.3 ms per frame (bus.jpg), about 80 fps model only |

Setup notes: python3-venv was missing and no sudo via scripted ssh, so the venv was created with `--without-pip` and pip bootstrapped from get-pip.py. The Ultralytics engine export pulled a torchvision that mismatched torch; fixed with `pip install --force-reinstall --no-deps torch==2.6.0 torchvision==0.21.0`.

## Test footage

Highway Vehicle Surveillance Dataset, Mendeley Data, CC BY 4.0. https://data.mendeley.com/datasets/zdgmdg7p25

| Clip | Resolution | fps | Length | View |
|------|-----------:|----:|-------:|------|
| day.mp4 | 3840x2160 | 30 | 5 s | Overhead, two lanes approaching, one SUV in left lane, strong tree shadows |
| night.mp4 | 1280x590 | 25 | 14 s | Elevated, divided highway, near carriageway approaching with headlights, far carriageway taillights |
| rain.mp4 | 704x432 | 30 | 6 s | Low res DOT style camera, wet road, near carriageway moving bottom-left to top-right, barrier wall |

Lane files are in `yolo/lanes/`. Polygons were set from single frames by inspection, not by the click tool.

## Run log, 2026-10-10

### Day clip, lane = right lane by the yellow edge line

- 1 alarm on track 1 entering the right lane at the top of frame. Correct.
- First run wrote no evidence clip: the writer waited for 5 s of post-event frames and the clip ended first. Fixed by flushing partial clips at end of stream. Rerun wrote a 149 frame clip plus JSON.
- 20 fps end to end. The limit is CPU decode and preprocess of 4K frames, not the engine.

### Rain clip, lane = lane along the concrete barrier

- Detector works well: 1.97 detections per frame, 16 tracks.
- First run: zero alarms. Diagnostic showed 5 tracks with 17 to 20 frames inside the polygon but negative speed along the crew direction. The direction guess was backwards.
- After flipping the direction: 1 alarm on track 2, evidence clip written. Other lane users fell inside the 3 s global cooldown of a 6 s clip, which is the intended platoon behaviour.
- 66 fps end to end at 704x432.

### Night clip, lane = right lane of near carriageway

Pretrained YOLO11n fails on this clip. This is the headlight glare risk from the design doc appearing in the first real test.

| Setting | Detections per frame | Tracks | Notes |
|---------|---------------------:|-------:|-------|
| conf 0.3, 640 px engine | 0.03 | 1 | Only a far carriageway taillight car, moving away |
| conf 0.1, 640 px engine | 0.06 | 1 | Same car, slightly longer track |
| conf 0.1, 1280 px, .pt weights | 0.08 | 4 | Three single frame detections plus the same far car |

No near carriageway vehicle was detected in any setting even though headlights are plainly visible to a human.

Model size and enhancement sweep, all at conf 0.1 and 640 px with .pt weights:

| Model | Plain, det/frame | Plain, inbound tracks | Enhanced, det/frame | Enhanced, inbound tracks |
|-------|-----------------:|----------------------:|--------------------:|-------------------------:|
| yolo11n | 0.06 | 0 | 0.07 | 0 |
| yolo11s | 0.07 | 0 | 0.03 | 0 |
| yolo11m | 0.09 | 1 (14 frames, left lane) | 0.01 | 1 (5 frames) |

Conclusions:

- Pretrained COCO weights do not see headlight-facing cars on very dark road at any size tested. Headlight bloom hides the vehicle body, and COCO contains almost no imagery of this kind.
- Gamma plus CLAHE made things worse for the two larger models. It amplifies the glare halo as much as the car.
- The medium model was the only one to produce an inbound near carriageway track, and it ran at a fraction of the speed.

What this means for the design: the night glare risk is confirmed as the top technical risk. Options, in order of cost: fine tune on night work zone footage with headlight glare, since even a few hundred labeled frames usually recover recall sharply; add a thermal camera, which removes the problem entirely; or require the work zone to be lit by light towers, which is normal practice on night closures and makes the scene far closer to the day case. The HVSD night clip is darker than a lit work zone would be, so it is a worst case rather than a typical one.

## Open items

- Collect or find night footage of a lit work zone to test the typical case rather than the worst case.
- Label a few hundred night frames and fine tune yolo11n on the 4060 Ti.
- Longer clips are needed for false alarm rate. Each HVSD clip is under 15 s.
- Record from the DelDOT public streams for multi minute footage.
- Measure true latency with the flashing LED method once a live camera is attached.
