import os
import cv2
import time
import threading
import queue
import numpy as np
import torch
from ultralytics import YOLO
import ctypes  # <-- added: get screen size on Windows

# --- NEW: display / perf tuning ---
DISPLAY_SCALE = 2.0        # phóng to khung hiển thị (2.0 = 200%)
# Nếu muốn FPS cao hơn: thử TARGET_WIDTH = 256, hoặc dùng model nhỏ hơn 'yolov8n.pt'
torch.set_num_threads(4)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# user config
VIDEO_SOURCE = 1            # webcam index or path
TARGET_WIDTH = 320         # smaller width -> faster
CONF = 0.5
IOU = 0.45

# choose device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Load (or replace with a smaller model to be faster, e.g. 'yolov8n.pt')
model = YOLO(r"D:\xiangqi_robot_TrainningAI_Final_2\models_chinesechess1\content\runs\detect\train\weights\best.pt")
if DEVICE == "cuda":
    try:
        model.to(DEVICE)
        # try half precision for speed
        try:
            model.model.half()
        except Exception:
            pass
    except Exception:
        print("Failed to move model to CUDA; falling back to CPU")
        DEVICE = "cpu"

cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)

# Create resizable window once
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)

# --- new: set initial window size based on screen resolution and DISPLAY_SCALE ---
try:
    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
except Exception:
    screen_w, screen_h = 1366, 768

# estimate capture height (fallback to 480)
cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if 'cap' in globals() and cap is not None else 480
initial_w = min(int(TARGET_WIDTH * DISPLAY_SCALE), max(200, screen_w - 200))
initial_h = min(int(cap_h * DISPLAY_SCALE), max(200, screen_h - 200))
cv2.resizeWindow("YOLOv8 Inference", initial_w, initial_h)

frame_q = queue.Queue(maxsize=1)
stop_event = threading.Event()

def capture_thread():
    while not stop_event.is_set():
        ret, frm = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        # keep only latest frame to avoid backlog
        if not frame_q.empty():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        try:
            frame_q.put_nowait(frm)
        except queue.Full:
            pass
    # cleanup
    try:
        cap.release()
    except Exception:
        pass

t = threading.Thread(target=capture_thread, daemon=True)
t.start()

prev_time = None
fps = 0.0
alpha = 0.2  # EMA smoothing for displayed FPS

try:
    while True:
        try:
            frame = frame_q.get(timeout=1.0)
        except queue.Empty:
            continue

        # resize preserving aspect ratio to target width
        h, w = frame.shape[:2]
        if w != TARGET_WIDTH:
            scale = TARGET_WIDTH / float(w)
            new_w = TARGET_WIDTH
            new_h = max(1, int(h * scale))
            inp = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            inp = frame

        # inference (pass device and smaller imgsz)
        try:
            results = model.predict(inp, conf=CONF, iou=IOU, device=DEVICE, imgsz=TARGET_WIDTH, verbose=False)
        except Exception as e:
            # fallback to CPU single-threaded predict if GPU fails
            try:
                results = model.predict(inp, conf=CONF, iou=IOU, imgsz=TARGET_WIDTH, verbose=False)
            except Exception as e2:
                print("Inference error:", e2)
                continue

        # draw boxes from results
        for result in results:
            for box in getattr(result, "boxes", []):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                name = model.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                cv2.rectangle(inp, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(inp, f"{name} {conf:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        # compute/display FPS (smoothed)
        now = time.time()
        if prev_time is None:
            cur_fps = 0.0
        else:
            dt = now - prev_time
            cur_fps = 1.0 / dt if dt > 1e-6 else 0.0
        prev_time = now
        fps = alpha * cur_fps + (1 - alpha) * fps
        cv2.putText(inp, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        # --- scale up for display (keeps inference image small but shows larger window) ---
        try:
            display_w = int(inp.shape[1] * DISPLAY_SCALE)
            display_h = int(inp.shape[0] * DISPLAY_SCALE)
            disp = cv2.resize(inp, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            disp = inp

        cv2.imshow("YOLOv8 Inference", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    stop_event.set()
    t.join(timeout=1.0)
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()

def find_external_opencv_index(max_idx=6):
    """Scan indices 0..max_idx-1, return first index that looks like external webcam
       (not robust but returns the first non-empty capture that's not the default if possible)."""
    good = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            good.append((i, w, h))
        cap.release()
    # prefer index > 0 (external) if available
    for idx, w, h in good:
        if idx != 0:
            return idx
    return good[0][0] if good else None

# Prefer RealSense (if installed)
VIDEO_MODE = os.environ.get("VIDEO_MODE", "auto")  # "auto" | "realsense" | "opencv"
VIDEO_INDEX = os.environ.get("VIDEO_INDEX")  # allow override

use_realsense = False
try:
    if VIDEO_MODE in ("auto", "realsense"):
        import pyrealsense2 as rs
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) > 0:
            # pick first RealSense D435* device
            chosen = None
            for d in devs:
                name = d.get_info(rs.camera_info.name) or ""
                serial = d.get_info(rs.camera_info.serial_number) or ""
                if "D435" in name or "D435i" in name:
                    chosen = (name, serial)
                    break
            if chosen is None:
                d = devs[0]
                chosen = (d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
            print("Using RealSense device:", chosen)
            use_realsense = True
            # if you use RealSense pipeline later, start it with cfg.enable_device(serial)
except Exception:
    use_realsense = False

if use_realsense:
    # keep using your pyrealsense2 pipeline logic (start_pipeline) and bind to device serial
    pass
else:
    # fallback to OpenCV; let user override via env or auto-scan
    if VIDEO_INDEX is not None:
        vid_idx = int(VIDEO_INDEX)
    else:
        # try to pick external camera (prefer index != 0)
        vid_idx = find_external_opencv_index(max_idx=6)
        if vid_idx is None:
            vid_idx = 0
    print(f"Opening OpenCV camera index: {vid_idx}")
    cap = cv2.VideoCapture(vid_idx, cv2.CAP_DSHOW)
    # optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
