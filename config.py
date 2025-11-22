# =============================================================================
# === FILE: config.py (CẤU HÌNH TOÀN HỆ THỐNG) ===
# =============================================================================

# --- THÔNG SỐ ROBOT & BÀN CỜ ---
# (Các tọa độ này sẽ được ghi đè bởi quy trình Hiệu chỉnh tự động, nhưng cứ để làm gốc)
BOARD_ORIGIN_X = 0.0
BOARD_ORIGIN_Y = 0.0
CELL_SIZE = 40.0

# Tọa độ bãi chứa quân bị ăn (X, Y, Z)
CAPTURE_BIN_X = -348.317
CAPTURE_BIN_Y = 324.831
CAPTURE_BIN_Z =311.929  # [QUAN TRỌNG] Độ cao khi thả quân vào thùng

# Độ cao an toàn (mm)
SAFE_Z  = 411.076  # Bay trên cao
PICK_Z  = 258.952  # Hạ xuống gắp (Cần đo thật chuẩn)
PLACE_Z = 271.116  # Hạ xuống đặt

# Cấu hình Kẹp (Gripper) - Tùy chỉnh theo loại van của bạn
GRIPPER_CLOSE = 1
GRIPPER_OPEN = 0
MOVE_SPEED = 50

# Góc xoay của đầu Robot (Rx, Ry, Rz)
ROTATION = [90.352,-1.147, -178.023] 

# Kết nối Robot
ROBOT_IP = "192.168.58.2"
DRY_RUN = False  # Đổi thành True nếu muốn test code mà không cần bật Robot

# --- THÔNG SỐ AI ---
AI_THINK_TIME = 10.0  # [QUAN TRỌNG] Thời gian suy nghĩ tối đa (giây)
AI_DEPTH = 3         # Độ sâu mặc định (sẽ bị ghi đè bởi logic tự động)

# Giá trị quân cờ (Dùng cho hàm đánh giá)
PIECE_VALUES = {
    'K': 10000, 'R': 100, 'C': 50, 'N': 45, 'E': 20, 'A': 20, 'P': 9
}

# Tọa độ về nhà (Home) để né Camera
IDLE_X = -35.82
IDLE_Y = 236.56
IDLE_Z = 336.679