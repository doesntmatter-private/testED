# Development setup on RTX 4060 Ti

Steps to get both pipeline options running before and after the Jetson arrives. Three development machines are covered:

| Machine | Option A DeepStream | Option B YOLO11n | Best use |
|---------|---------------------|------------------|----------|
| RTX 4060 Ti, Ubuntu 24.04 native | Yes | Yes | Full desktop development, model training |
| RTX 4060 Ti, Windows 11 native | No, WSL2 only | Yes | Option B prototype without dual boot |
| Jetson Orin Nano Super | Yes | Yes | Primary development once it arrives. All timing numbers are real |

Sections 1 to 8 cover Ubuntu on the desktop. Section 9 covers native Windows 11. Section 10 covers developing directly on the Jetson. WSL2 is at the end as a fallback.

DeepStream version choice: **7.1**. It matches what JetPack 6.2 ships on Orin Nano, so configs and engines rebuild without surprises. DeepStream 8.0 also works on Ada but targets JetPack 7 on the Jetson side.

## 0. Hardware check

- RTX 4060 Ti seated, display on the GPU or on integrated graphics, either is fine.
- 32 GB RAM recommended, 16 GB works.
- 100 GB free disk. The DeepStream container is about 15 GB, models and video add more.
- Wired Ethernet if you plan to test an RTSP camera later.

## 1. Install Ubuntu 24.04

1. Download Ubuntu 24.04 LTS desktop ISO and write it to a USB stick.
2. Install. Choose the normal installation. Do not tick "install third party drivers" so you control the NVIDIA driver version yourself.
3. After first boot:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y build-essential git curl wget vim htop net-tools \
    python3-venv python3-pip ffmpeg v4l-utils
```

## 2. NVIDIA driver

DeepStream 7.1 is validated against the R560 driver. Newer 570 and 580 drivers also work for Ada.

```bash
sudo apt install -y nvidia-driver-560-open
sudo reboot
```

Verify:

```bash
nvidia-smi
```

You should see the 4060 Ti and a driver version 560 or higher. If the open kernel module gives trouble, swap to `nvidia-driver-560` without the `-open` suffix.

Disable automatic driver upgrades so an unattended update does not break the container pairing:

```bash
sudo apt-mark hold nvidia-driver-560-open
```

## 3. Docker and the NVIDIA Container Toolkit

```bash
# Docker
sudo apt install -y ca-certificates gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Log out and back in so the docker group applies, then verify GPU passthrough:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

## 4. Project layout

```bash
mkdir -p ~/src/jetson/{data/video,data/models,deepstream,yolo}
cd ~/src/jetson
```

- `data/video` holds test clips. Mounted into containers.
- `data/models` holds ONNX files and per GPU engine files. Engines are not portable between GPUs, so keep them named by GPU.
- `deepstream` holds Option A configs and the probe app.
- `yolo` holds Option B Python code.

## 5. Get test video

You need footage of a road from a fixed elevated camera before anything else is useful. Options in order of preference:

1. Record 30 to 60 minutes from an overpass or shoulder with a phone on a tripod at 1080p30. Keep the camera still.
2. Public traffic camera archives. Many state DOTs publish RTSP or HLS feeds. Record with ffmpeg:

```bash
ffmpeg -i "https://example-dot-feed/stream.m3u8" -t 600 -c copy data/video/dot_sample.mp4
```

3. The DeepStream container ships `sample_1080p_h264.mp4`, which is enough for pipeline smoke tests.

Convert anything that is not H.264 so both pipelines decode it on NVDEC:

```bash
ffmpeg -i input.mov -c:v libx264 -preset fast -crf 20 -r 30 -an data/video/workzone_01.mp4
```

## 6. Option A: DeepStream 7.1

### 6.1 Pull and start the container

```bash
docker pull nvcr.io/nvidia/deepstream:7.1-triton-multiarch

xhost +local:docker   # only if you want an on screen display
docker run --gpus all -it --rm --net=host --name ds71 \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/src/jetson:/work \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch
```

If the pull asks for credentials, create a free NGC account, generate an API key, and run `docker login nvcr.io` with username `$oauthtoken` and the key as the password.

### 6.2 Smoke test inside the container

```bash
cd /opt/nvidia/deepstream/deepstream
./user_additional_install.sh      # installs extra GStreamer plugins
cd samples/configs/deepstream-app
deepstream-app -c source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt
```

Four tiled streams with boxes should appear. First run compiles TensorRT engines and takes a few minutes. If you have no display, edit the config and set `[sink0] type=1` for a fake sink, or `type=3` with a file path.

### 6.3 Run the analytics sample

This is the sample closest to the design. It already does ROI entry, line crossing and direction.

```bash
cd /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-nvdsanalytics-test
export CUDA_VER=12.6
make
./deepstream-nvdsanalytics-test file:///work/data/video/workzone_01.mp4
```

The terminal prints object counts and ROI membership per frame. The ROI and direction config lives in `config_nvdsanalytics.txt` in the same folder. Edit `roi-RF` coordinates to your closed lane polygon and `direction-South` to point at the crew.

### 6.4 Pull TrafficCamNet

```bash
pip3 install nvidia-pyindex ngc-cli 2>/dev/null || true
mkdir -p /work/data/models/trafficcamnet && cd /work/data/models/trafficcamnet
wget --content-disposition \
  'https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet/versions/pruned_v1.0.4/zip' -O tcn.zip
unzip tcn.zip
```

Then point an nvinfer config at it. Start from `samples/configs/tao_pretrained_models/config_infer_primary_trafficcamnet.txt` in the container and change `tlt-encoded-model` and `labelfile-path` to the downloaded files.

### 6.5 Python bindings for the probe

The intrusion logic is easier to write in Python than C.

```bash
cd /opt/nvidia/deepstream/deepstream/sources
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
cd deepstream_python_apps
git checkout v1.2.0      # matches DeepStream 7.1
pip3 install ./bindings/pyds-*.whl 2>/dev/null || \
  (cd bindings && ./build.sh && pip3 install ./pyds-*.whl)
cd apps/deepstream-nvdsanalytics
python3 deepstream_nvdsanalytics.py file:///work/data/video/workzone_01.mp4
```

Copy that script into `/work/deepstream/` as the starting point for the real app. The probe function `nvanalytics_src_pad_buffer_probe` is where the per track ROI status arrives.

### 6.6 Persist the container

Working in `--rm` containers loses installed packages. Once the setup is right, commit it:

```bash
docker commit ds71 jetson-ds:7.1-dev
```

## 7. Option B: YOLO11n in Python

Run this on the host, not in the container.

### 7.1 Environment

```bash
cd ~/src/jetson/yolo
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics shapely opencv-python lap
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 7.2 Export the TensorRT engine

```bash
yolo export model=yolo11n.pt format=engine half=True imgsz=640 device=0
mv yolo11n.engine ../data/models/yolo11n_4060ti_fp16.engine
```

Takes 2 to 5 minutes. Rebuild on the Jetson later. Do not copy this engine to the Jetson.

### 7.3 First tracking run

```bash
yolo track model=../data/models/yolo11n_4060ti_fp16.engine \
  source=../data/video/workzone_01.mp4 classes=2,3,5,7 \
  tracker=bytetrack.yaml show=True
```

You should see stable IDs on cars and trucks. Press q to quit.

### 7.4 Hardware decode in OpenCV (optional on desktop, required on Jetson)

The pip OpenCV wheel has no GStreamer support. On the desktop, CPU decode of one 1080p stream is fine, so skip this. On the Jetson, use the system `python3-opencv`, which is built with GStreamer, and open the capture with the GStreamer string in the design doc.

### 7.5 Lane polygon tool

Write a small script that shows the first frame, lets you click four points, and saves `lane.json`. Use it on your test video so the intrusion logic has something to test against.

### 7.6 Intrusion app skeleton

```
yolo/
  app.py           main loop: capture, track, intrusion, alert
  intrusion.py     IntrusionState class, pure Python, no CV deps
  polygon_tool.py  click to define lane.json
  alerts.py        GPIO relay on Jetson, print() on desktop
  tests/test_intrusion.py
```

Make `alerts.py` check for `Jetson.GPIO` and fall back to logging so the same code runs on both machines.

## 8. Measure before the Jetson arrives

On the 4060 Ti these numbers are not representative of Orin Nano, but they validate correctness:

- Run both pipelines on the same clip and compare track counts and ROI events frame by frame.
- Run the intrusion state machine unit tests against synthetic tracks: straight through, late merge out, stopped at taper, platoon.
- Record a clip with a known event and check that the saved evidence clip spans 5 seconds before and after.

## 9. Native Windows 11 on the 4060 Ti

Option B runs natively on Windows. DeepStream does not. If the 4060 Ti lives in a Windows machine you do not want to dual boot, develop Option B here and leave DeepStream until the Jetson arrives.

### 9.1 Option B setup

Install the latest GeForce driver, then Python 3.11 from python.org with "Add to PATH" ticked.

```powershell
mkdir C:\src\jetson\yolo; cd C:\src\jetson\yolo
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics shapely opencv-python lap
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
yolo export model=yolo11n.pt format=engine half=True imgsz=640 device=0
yolo track model=yolo11n.engine source=..\data\video\workzone_01.mp4 classes=2,3,5,7 tracker=bytetrack.yaml show=True
```

Ultralytics downloads the TensorRT wheels automatically on first export.

### 9.2 What differs from Linux

- **Decode.** The pip OpenCV wheel uses Media Foundation or FFmpeg on the CPU. Fine for one 1080p file or RTSP stream on a desktop. Use `cv2.VideoCapture(path)` with no GStreamer string.
- **Evidence encode.** No `nvv4l2h264enc`. Use OpenCV's VideoWriter, or shell out to `ffmpeg -c:v h264_nvenc`.
- **GPIO.** `Jetson.GPIO` does not exist. The `alerts.py` stub must log instead. Write it as:

```python
try:
    import Jetson.GPIO as GPIO
    HAVE_GPIO = True
except ImportError:
    HAVE_GPIO = False
```

- **Paths.** Use `pathlib.Path` everywhere so lane.json and clip paths work on both OSes.

### 9.3 Option A on Windows

Not available natively. Use the WSL2 fallback at the end of this document, and accept software decode. Do not measure latency there.

## 10. Developing directly on the Jetson Orin Nano Super

The Orin Nano Super is a full Ubuntu 22.04 desktop with a 6 core ARM CPU. Most teams do their primary development on it, and every fps and latency number you observe is real. The desktop GPU is a convenience for training and bulk video processing, not a requirement.

### 10.1 Which Nano

- **Orin Nano or Orin Nano Super.** Everything in the design doc works. JetPack 6.2, DeepStream 7.1, TensorRT 10, Python 3.10.
- **Original Jetson Nano from 2019.** Do not use. JetPack 4.6, Ubuntu 18.04, Python 3.6, TensorRT 8.2, Maxwell GPU, end of life. YOLO11 and DeepStream 7 do not run on it.

### 10.2 Flash and first boot

1. Download the JetPack 6.2 SD card image for the Orin Nano dev kit and write it with Balena Etcher. This is faster than SDK Manager and needs no Ubuntu host.
2. Boot once with monitor and keyboard, accept the license, create a user, connect to the network.
3. Install an NVMe drive in the M.2 slot. The SD card is slow for compiles and video buffers. Either reflash to NVMe with SDK Manager from an Ubuntu host, or clone the running SD install to NVMe:

```bash
sudo apt install -y git
git clone https://github.com/jetsonhacks/rootOnNVMe.git
cd rootOnNVMe && ./copy-rootfs-ssd.sh && ./setup-service.sh
sudo reboot
```

4. Set the power mode and clocks so you measure the real ceiling:

```bash
sudo nvpmodel -m 2          # MAXN SUPER on Orin Nano Super, check `sudo nvpmodel -q` for the index
sudo jetson_clocks
```

5. Install monitoring and update:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo pip3 install jetson-stats
sudo reboot
jtop
```

### 10.3 Remote development from the Mac

After first boot the monitor is not needed.

```bash
# on the Mac
ssh-copy-id user@jetson.local
```

In VS Code install the Remote SSH extension and connect to `user@jetson.local`. Edit, run, and debug as if local. Set `sudo systemctl set-default multi-user.target` to boot headless and free about 1 GB of RAM.

### 10.4 Option B on the Jetson

PyTorch is not a plain pip install on Jetson. Use NVIDIA's Jetson wheel index. The `--system-site-packages` flag matters: it gives the venv the apt OpenCV, which is built with GStreamer and is what enables hardware decode.

```bash
sudo apt install -y python3-pip python3-venv python3-opencv libopenblas-dev
mkdir -p ~/src/jetson && cd ~/src/jetson
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
pip install ultralytics shapely lap
python -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer   # must say YES
yolo export model=yolo11n.pt format=engine half=True imgsz=640 device=0
```

The engine export takes 5 to 15 minutes. Keep the resulting file; it is specific to this board.

Alternative: the Ultralytics Jetson container has all of this preinstalled.

```bash
docker run -it --runtime nvidia --ipc=host --network host \
  -v ~/src/jetson:/work ultralytics/ultralytics:latest-jetson-jetpack6
```

Capture on the Jetson uses the GStreamer string so decode stays on NVDEC:

```python
pipeline = (
  "rtspsrc location=rtsp://cam/stream latency=0 ! rtph264depay ! h264parse "
  "! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert "
  "! video/x-raw,format=BGR ! appsink drop=true max-buffers=1")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
```

For a file source replace `rtspsrc ... h264parse` with `filesrc location=/path.mp4 ! qtdemux ! h264parse`.

### 10.5 Option A on the Jetson

DeepStream 7.1 is included in JetPack 6.2 under `/opt/nvidia/deepstream/deepstream`. No container needed. The steps in sections 6.2 to 6.5 apply with two changes:

- `export CUDA_VER=12.6` is the same, but builds take 3 to 5 times longer.
- Use the prebuilt Python bindings wheel from the deepstream_python_apps GitHub release page instead of compiling. Compiling on the board takes about 20 minutes.

```bash
cd /opt/nvidia/deepstream/deepstream/sources
sudo git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
cd deepstream_python_apps && sudo git checkout v1.2.0
# download pyds-1.2.0-cp310-cp310-linux_aarch64.whl from the release page, then
pip3 install ./pyds-1.2.0-*.whl
```

### 10.6 GPIO and the relay HAT

```bash
sudo pip3 install Jetson.GPIO
sudo groupadd -f gpio && sudo usermod -aG gpio $USER
sudo cp /opt/nvidia/jetson-gpio/etc/99-gpio.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Test with a relay on pin 7:

```python
import Jetson.GPIO as GPIO, time
GPIO.setmode(GPIO.BOARD); GPIO.setup(7, GPIO.OUT)
GPIO.output(7, GPIO.HIGH); time.sleep(1); GPIO.output(7, GPIO.LOW)
```

### 10.7 What is slower or awkward on the board

- **Memory.** 8 GB is shared between CPU and GPU. Run headless and close heavy processes while measuring.
- **Engine builds.** 5 to 15 minutes versus 2 on a desktop. Build once and cache.
- **Training.** Do not fine tune on the Jetson. Collect frames here, train on the 4060 Ti or a cloud GPU, copy the ONNX back, and rebuild the engine.
- **Thermals.** The dev kit fan is adequate. Watch jtop during a long run; sustained GPU temperature above 80 °C means throttling.

### 10.8 Moving work from the desktop

- Copy the `deepstream/` and `yolo/` directories and the ONNX or .etlt model files. Leave engine files behind.
- First run of each pipeline rebuilds engines. Allow 10 minutes.
- Re-tune confirmation frames, inference size, and tracker interval here. Desktop values are meaningless on this board.

## 10b. Interim development on an NVIDIA T400

The T400 is a 30 W Turing card with 384 CUDA cores and 2 GB or 4 GB. TensorRT 10 and DeepStream 7.1 support it. Treat it as a slow single stream development card.

- Install the same driver 560 and follow sections 1 to 3 unchanged.
- Option B: follow section 7. Expect 30 to 50 fps for YOLO11n FP16 at 640 px, which is close to Orin Nano, so confirmation and heading tuning done here roughly carries over.
- Option A: follow section 6. One stream on the 2 GB card, two or three on the 4 GB card. Check `nvidia-smi` memory on first run.
- No NVENC. Encode evidence clips on the CPU, and record any rolling buffer as the raw RTSP stream without re-encoding.
- Do not train on it.

The scaffolded Option B code lives in `yolo/`. Its README has the exact commands for the T400.

## 10c. Runbook: commands used on the T400 box

Run these in a shell on the Linux box, from `~/src/jetson` unless a `cd` says otherwise. The box in this project is `$USER@10.111.72.128` with an NVIDIA T400 4GB, driver 580, Python 3.10, Ubuntu 22.04. Each gotcha we hit is noted where it applies.

### Get the repo onto the box

From the Mac, one time:

```bash
rsync -av --exclude '.venv' --exclude '__pycache__' --exclude '.DS_Store' \
  ~/src/pers/jetson/ $USER@10.111.72.128:/home/$USER/src/jetson/
```

Then on the box, confirm the environment:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python3 --version
lsb_release -ds
df -h /home | tail -1
```

### Python environment

If `python3 -m venv .venv` fails with "ensurepip is not available", either install the package or bootstrap pip by hand.

```bash
# clean route, needs sudo
sudo apt install -y python3.10-venv python3-pip

# no sudo route
cd ~/src/jetson/yolo
python3 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
```

Then install and verify:

```bash
cd ~/src/jetson/yolo
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -c "import torch, ultralytics, cv2; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(ultralytics.__version__, cv2.__version__)"
```

Every later command assumes the venv is active: `source ~/src/jetson/yolo/.venv/bin/activate`.

### TensorRT engine export

About 7 minutes on the T400, 68 MB result. Engines are GPU specific; never copy them between machines.

```bash
cd ~/src/jetson/yolo
yolo export model=yolo11n.pt format=engine half=True imgsz=640 device=0
mv yolo11n.engine ../data/models/yolo11n_t400_fp16.engine
```

Gotcha: the export pulled a torchvision that did not match torch, and the first inference failed with `operator torchvision::nms does not exist`. Fix and verify:

```bash
pip install --force-reinstall --no-deps torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
yolo predict model=../data/models/yolo11n_t400_fp16.engine source=https://ultralytics.com/images/bus.jpg device=0 save=False
# expect a Speed line near: 14.4ms preprocess, 12.3ms inference, 6.7ms postprocess
```

### Test footage

The three clips `day.mp4`, `night.mp4` and `rain.mp4` come from the **Highway Vehicle Surveillance Dataset (HVSD)** published on Mendeley Data in 2025, DOI 10.17632/zdgmdg7p25.1, at https://data.mendeley.com/datasets/zdgmdg7p25. It is licensed **CC BY 4.0**, so it can be used freely with attribution to the dataset authors. The archive `Dataset.zip` is 41 MB and unpacks to `Dataset/day.mp4` (3840x2160, 30 fps, 5 s), `Dataset/night.mp4` (1280x590, 25 fps, 14 s) and `Dataset/rain.mp4` (704x432, 30 fps, 6 s). The same record also offers DAY, NIGHT and RAINY experiment archives with annotations, which we have not used. Each clip is under 15 seconds, so they are smoke test material, not enough for false alarm statistics.

```bash
mkdir -p ~/src/jetson/data/video/hvsd && cd ~/src/jetson/data/video/hvsd
curl -sSL -o Dataset.zip "https://data.mendeley.com/public-files/datasets/zdgmdg7p25/files/27f999f6-8c95-4aa8-8955-1071808ff681/file_downloaded"
unzip -oq Dataset.zip
find . -iname "*.mp4" -exec ls -la {} \;
```

Print clip properties and save one frame from each for drawing the polygon:

```bash
cd ~/src/jetson && mkdir -p data/video/frames
python - <<'PY'
import cv2
for n in ["day","night","rain"]:
    p=f"data/video/hvsd/Dataset/{n}.mp4"; c=cv2.VideoCapture(p)
    w,h,fps,nf=int(c.get(3)),int(c.get(4)),c.get(5),int(c.get(7))
    print(f"{n}: {w}x{h} {fps:.1f}fps {nf} frames {nf/fps:.0f}s")
    c.set(1, min(100,nf-1)); ok,f=c.read()
    if ok: cv2.imwrite(f"data/video/frames/{n}.jpg", f)
PY
```

To look at the frames from the Mac, pull them over. Quote the remote glob or zsh expands it locally:

```bash
scp "$USER@10.111.72.128:/home/$USER/src/jetson/data/video/frames/*.jpg" ~/Downloads/
```

### Sync edits made on the Mac

From the Mac. Copy folders that have subdirectories with `-r`; `scp a.py static/index.html host:dir/` flattens both into `dir/`.

```bash
scp -r ~/src/pers/jetson/yolo/laneguard ~/src/pers/jetson/yolo/lanes $USER@10.111.72.128:/home/$USER/src/jetson/yolo/
```

### Run the app headless on a clip

```bash
cd ~/src/jetson/yolo
for n in night day rain; do
  echo "=== $n ==="
  python -m laneguard.app --source ../data/video/hvsd/Dataset/$n.mp4 --lane lanes/hvsd_$n.json \
    --model ../data/models/yolo11n_t400_fp16.engine --evidence ../data/events/$n \
    2>&1 | grep -E "ALARM|near miss|processed|wrote|Traceback"
done
find ../data/events -type f | sort
```

### Diagnose a lane file against a clip

Prints every track with frames inside the polygon and inbound run length, and saves an overlay image with trajectories. This is what caught the backwards crew direction on the rain clip.

```bash
cd ~/src/jetson/yolo
python -m laneguard.diag --source ../data/video/hvsd/Dataset/rain.mp4 --lane lanes/hvsd_rain.json \
  --model ../data/models/yolo11n_t400_fp16.engine \
  --save-overlay ../data/video/frames/rain_overlay.jpg --overlay-frame 100
```

Model and enhancement sweep used on the night clip. The `.pt` weights download on first use.

```bash
for m in yolo11n.pt yolo11s.pt yolo11m.pt; do
  for e in "" "--enhance"; do
    echo "=== $m ${e:-plain} ==="
    python -m laneguard.diag --source ../data/video/hvsd/Dataset/night.mp4 --lane lanes/hvsd_night.json \
      --model $m --conf 0.1 $e 2>&1 | grep -E "detections/frame|^ +[0-9]+ +[0-9]"
  done
done
```

### Run the app with the web interface

Start detached so it survives logout, loop the clip, serve on port 8080. Then open `http://10.111.72.128:8080` from any browser on the network.

```bash
cd ~/src/jetson/yolo
pip install fastapi uvicorn
nohup python -m laneguard.app --source ../data/video/hvsd/Dataset/rain.mp4 --lane lanes/hvsd_rain.json \
  --model ../data/models/yolo11n_t400_fp16.engine --evidence ../data/events/web --web 8080 --loop \
  > ../data/laneguard.log 2>&1 < /dev/null &
disown
sleep 10; tail -3 ../data/laneguard.log
curl -s http://127.0.0.1:8080/api/state | python3 -m json.tool | head -30
```

Operate the running instance:

```bash
tail -f ~/src/jetson/data/laneguard.log                 # follow the log
pgrep -af "[l]aneguard.app"                             # is it running
pkill -f "[l]aneguard.app"                              # stop it
curl -s "http://127.0.0.1:8080/api/lcs/closures?district=4&status=active&lane_only=true&limit=3"   # Caltrans feed reachable?
```

Gotcha: `pkill -f "laneguard.app"` also matches the shell that is running the pkill, because the pattern appears in its own command line. The bracket form `"[l]aneguard.app"` matches the Python process but not itself. This matters most when stop and start are chained in one line.

Edits to `static/index.html` take effect on browser reload. Edits to any `.py` file need a stop and start.

### ffmpeg without sudo

The box had no ffmpeg. The johnvansickle static build installed but `ffprobe` segfaulted on an HTTPS HLS stream. The BtbN build is the replacement.

```bash
mkdir -p ~/.local/bin ~/tmp && cd ~/tmp
curl -sSL -o ffbtbn.tar.xz https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
tar xf ffbtbn.tar.xz
cp ffmpeg-master-latest-linux64-gpl/bin/ffmpeg ffmpeg-master-latest-linux64-gpl/bin/ffprobe ~/.local/bin/
rm -rf ffmpeg-master-latest-linux64-gpl ffbtbn.tar.xz
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
ffprobe -version | head -1
```

ffmpeg tools take single dash options: `-show_entries`, not `--show_entries`.

### Probe and record public traffic streams

Caltrans lists its cameras per district with lat/lon, route and a `streamingVideoURL` at `https://cwwp2.dot.ca.gov/data/d4/cctv/cctvStatusD04.json`. DelDOT streams follow `http://video.deldot.gov:1935/live/KCAM00N.stream/playlist.m3u8`. Individual Caltrans cameras come and go, and most are low resolution.

```bash
mkdir -p ~/src/jetson/data/video/streams && cd ~/src/jetson/data/video/streams

# probe
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate -of csv=p=0 \
  "https://wzmedia.dot.ca.gov/D4/W4_at_Lone_Tree_Way.stream/playlist.m3u8"
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate -of csv=p=0 \
  "http://video.deldot.gov:1935/live/KCAM003.stream/playlist.m3u8"

# record 10 minutes without re-encoding, in the background
nohup ffmpeg -v error -i "http://video.deldot.gov:1935/live/KCAM003.stream/playlist.m3u8" \
  -t 600 -c copy -bsf:a aac_adtstoasc deldot_us13_kcam003.mp4 < /dev/null > deldot_us13_kcam003.log 2>&1 &
disown
```

Then run the app or the diagnostic on the recording the same way as the HVSD clips.

### Working from the Mac instead

Any command above can be wrapped as `ssh $USER@10.111.72.128 '<command>'`. When driving the box through Claude Code, prefix the whole ssh command with `!` so it runs in the session and Claude sees the output; a managed policy in this environment prevents Claude from running ssh or scp itself. Note that scripted ssh has no terminal, so `sudo` cannot prompt for a password there.

## 11. Recommended split once everything is in hand

| Task | Machine |
|------|---------|
| Pipeline development, intrusion logic, polygon tool | Jetson, via VS Code remote |
| Latency and fps measurement, tuning | Jetson only |
| Model fine tuning | 4060 Ti |
| Bulk processing of recorded video to find events | 4060 Ti |
| Unit tests for pure Python logic | Anywhere, including the Mac |

## Fallback: WSL2 on Windows 11

Works for DeepStream inference and logic development. Hardware decode through `nvv4l2decoder` is unreliable, so use software decode:

1. Install the Windows NVIDIA driver 560 or newer. Do not install a driver inside WSL.
2. `wsl --install -d Ubuntu-24.04`, then install Docker Desktop with the WSL2 backend and enable GPU support.
3. Run the same DeepStream container. In pipeline configs replace `nvv4l2decoder` with `avdec_h264 ! nvvideoconvert`.
4. Option B works unchanged in WSL2, but it is simpler to run it natively on Windows per section 9.

Switch to native Ubuntu or the Jetson before measuring latency.
