import os
import cv2
import time
import threading
import queue
import numpy as np
import torch
from ultralytics import YOLO
import ctypes

# ==========================================
# 1. CẤU HÌNH
# ==========================================
DISPLAY_SCALE = 1.5         # Zoom khung hình hiển thị
TARGET_WIDTH = 320          # Resize ảnh nhỏ để AI chạy nhanh
CONF = 0.5
IOU = 0.45
# Đường dẫn model của bạn
MODEL_PATH = r"D:\xiangqi_robot_TrainningAI_Final_4\models_chinesechess1\content\runs\detect\train\weights\best.pt"

# ==========================================
# 2. HÀM HỖ TRỢ TÌM CAMERA (Đưa lên đầu)
# ==========================================
def find_external_opencv_index(max_idx=4):
    """Quét tìm camera (ưu tiên camera ngoài, index khác 0)"""
    print("🔄 Đang quét tìm Camera...")
    good_cam = 0
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"   - Tìm thấy Camera index {i}")
                # Ưu tiên lấy camera không phải 0 (thường 0 là webcam laptop)
                if i > 0: 
                    good_cam = i
                    cap.release()
                    break 
            cap.release()
    return good_cam

# ==========================================
# 3. KHỞI TẠO CAMERA & DEVICE
# ==========================================
# Chọn Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Using device:", DEVICE)

# Xử lý chọn Camera (Logic RealSense hoặc OpenCV)
use_realsense = False
pipeline = None
cap = None

# Kiểm tra RealSense trước (Nếu bạn có cài thư viện và cắm thiết bị)
try:
    import pyrealsense2 as rs
    ctx = rs.context()
    if len(ctx.query_devices()) > 0:
        print("📷 Phát hiện RealSense! Đang khởi tạo...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        use_realsense = True
        print("✅ RealSense đã sẵn sàng.")
except:
    use_realsense = False

# Nếu không có RealSense thì dùng OpenCV
if not use_realsense:
    # Ưu tiên lấy từ biến môi trường, nếu không thì tự quét
    env_idx = os.environ.get("VIDEO_INDEX")
    if env_idx:
        vid_idx = int(env_idx)
    else:
        vid_idx = find_external_opencv_index()
    
    print(f"📷 Đang mở Camera OpenCV Index: {vid_idx}")
    cap = cv2.VideoCapture(vid_idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==========================================
# 4. LOAD MODEL
# ==========================================
print("🧠 Đang load Model YOLO...")
try:
    model = YOLO(MODEL_PATH)
    if DEVICE == "cuda":
        model.to(DEVICE)
        try: model.model.half() # Tăng tốc bằng FP16
        except: pass
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    # Nếu lỗi đường dẫn thì load model chuẩn để test tạm
    print("⚠️ Đang load yolov8n.pt để test tạm...")
    model = YOLO("yolov8n.pt")

# ==========================================
# 5. MULTI-THREADING CAPTURE
# ==========================================
frame_q = queue.Queue(maxsize=1)
stop_event = threading.Event()

def capture_thread():
    """Luồng đọc camera riêng biệt để tăng FPS"""
    global cap, pipeline
    while not stop_event.is_set():
        frm = None
        if use_realsense:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    frm = np.asanyarray(color_frame.get_data())
            except: pass
        else:
            if cap and cap.isOpened():
                ret, img = cap.read()
                if ret: frm = img
            else:
                time.sleep(0.1) # Chờ nếu mất kết nối
        
        if frm is not None:
            # Chỉ giữ frame mới nhất
            if not frame_q.empty():
                try: frame_q.get_nowait()
                except: pass
            frame_q.put(frm)
        else:
            time.sleep(0.01)
    
    # Cleanup
    if use_realsense and pipeline: pipeline.stop()
    if cap: cap.release()

# Bắt đầu luồng
t = threading.Thread(target=capture_thread, daemon=True)
t.start()

# ==========================================
# 6. VÒNG LẶP CHÍNH (MAIN LOOP)
# ==========================================
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)

# Lấy kích thước màn hình để chỉnh cửa sổ
try:
    user32 = ctypes.windll.user32
    screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
except: screen_w, screen_h = 1366, 768

initial_w = min(int(TARGET_WIDTH * DISPLAY_SCALE), screen_w - 100)
initial_h = min(int(480 * DISPLAY_SCALE), screen_h - 100)
cv2.resizeWindow("YOLOv8 Inference", initial_w, initial_h)

prev_time = time.time()
fps = 0.0
alpha_fps = 0.2

print("\n=== ĐANG CHẠY (Bấm 'Q' để thoát) ===")

try:
    while True:
        # 1. Lấy ảnh
        try:
            frame = frame_q.get(timeout=1.0)
        except queue.Empty:
            continue

        # 2. Resize đầu vào cho AI (giữ nguyên tỉ lệ)
        h, w = frame.shape[:2]
        scale = TARGET_WIDTH / float(w)
        inp = cv2.resize(frame, (TARGET_WIDTH, int(h * scale)))

        # 3. Predict
        results = model.predict(inp, conf=CONF, iou=IOU, verbose=False, imgsz=TARGET_WIDTH)

        # 4. Vẽ
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                name = model.names.get(cls_id, str(cls_id))
                conf_score = float(box.conf[0])
                
                # Màu: Đỏ (Red) hoặc Xanh (Black) - Giả sử tên có 'r_' là đỏ
                color = (0, 0, 255) if "r_" in name else (0, 255, 0)
                cv2.rectangle(inp, (x1, y1), (x2, y2), color, 2)
                cv2.putText(inp, f"{name} {conf_score:.2f}", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 5. FPS
        curr_time = time.time()
        fps = fps * (1 - alpha_fps) + (1.0 / (curr_time - prev_time)) * alpha_fps
        prev_time = curr_time
        cv2.putText(inp, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 6. Hiển thị (Zoom to)
        disp = cv2.resize(inp, (int(inp.shape[1]*DISPLAY_SCALE), int(inp.shape[0]*DISPLAY_SCALE)))
        cv2.imshow("YOLOv8 Inference", disp)

        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt: pass
finally:
    stop_event.set()
    t.join()
    cv2.destroyAllWindows()