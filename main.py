# ===================================================================================
# === FILE: main.py (FIXED: DEBUG GRID ADDED) ===
# ===================================================================================
import sys
import os
import time
import numpy as np
import cv2
import traceback
import json
import math
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
from ultralytics import YOLO

import config
import xiangqi
import ai

# IMPORT BOOK
try:
    import ai_book 
except ImportError:
    try:
        import opening_book as ai_book
    except ImportError:
        print("❌ LỖI: Không tìm thấy file sách (ai_book.py)")
        sys.exit()

from robot import FR5Robot
import sound_player

# ==========================================
# 0. CẤU HÌNH CHẾ ĐỘ (CONFIG)
# ==========================================
# Đã fix cứng thành False để dùng Camera
ALLOW_MOUSE_MOVE = False 

print(f"\n=== CHẾ ĐỘ ĐIỀU KHIỂN: {'🖱️ DÙNG CHUỘT (DEBUG)' if ALLOW_MOUSE_MOVE else '📷 DÙNG CAMERA (REAL)'} ===")

# ==========================================
# 1. KHỞI TẠO
# ==========================================
pygame.init()
pygame.font.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SQUARE_SIZE = 40
NUM_COLS = xiangqi.NUM_COLS
NUM_ROWS = xiangqi.NUM_ROWS
BOARD_WIDTH = (NUM_COLS - 1) * SQUARE_SIZE
START_X = (SCREEN_WIDTH - BOARD_WIDTH) / 2
START_Y = (SCREEN_HEIGHT - ((NUM_ROWS - 1) * SQUARE_SIZE)) / 2 - 20
LINE_COLOR = (0, 0, 0)
BOARD_COLOR = (252, 230, 201)
PIECE_RADIUS = SQUARE_SIZE // 2 - 4
PIECE_FONT = pygame.font.SysFont('simsun', 20, bold=True)
GAME_FONT = pygame.font.SysFont('times new roman', 36, bold=True)
UI_FONT = pygame.font.SysFont('arial', 16, bold=True)

BTN_SURRENDER_RECT = pygame.Rect(SCREEN_WIDTH/2 - 60, SCREEN_HEIGHT - 60, 120, 40)
BTN_COLOR = (200, 50, 50)

PIECE_DISPLAY_NAMES = {
    'r_K': '帥', 'r_A': '仕', 'r_E': '相', 'r_R': '俥', 'r_N': '傌', 'r_C': '炮', 'r_P': '兵',
    'b_K': '將', 'b_A': '士', 'b_E': '象', 'b_R': '車', 'b_N': '馬', 'b_C': '砲', 'b_P': '卒'
}

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(f"Xiangqi Robot - Mode: {'MOUSE' if ALLOW_MOUSE_MOVE else 'CAMERA'}")

# ==========================================
# 2. TRẠNG THÁI GAME
# ==========================================
board = xiangqi.get_board()
turn = 'r'
game_over = False
winner = None
last_move = None
selected_pos = None 
r_captured = []
b_captured = []
move_history = []

# ==========================================
# 3. ROBOT & CAMERA CONFIG
# ==========================================
robot = FR5Robot()
try:
    robot.connect()
except Exception as e:
    print(f"[MAIN] Lỗi kết nối Robot: {e}")
    if not config.DRY_RUN: sys.exit()
    print("[MAIN] Đang chạy chế độ DRY_RUN.")

# --- HIỆU CHỈNH ROBOT ---
print("\n--- HIỆU CHỈNH ROBOT ---")
dst_pts_logic = np.array([[0, 0], [8, 0], [8, 9], [0, 9]], dtype=np.float32)
try:
    if config.DRY_RUN:
        src_pts_fake = np.array([[200, -100], [520, -100], [520, 260], [200, 260]], dtype=np.float32)
        robot.set_perspective_matrix(cv2.getPerspectiveTransform(dst_pts_logic, src_pts_fake))
    else:
        print("Đang đọc tọa độ từ Robot...")
        err1, data1 = robot.robot.GetRobotTeachingPoint("goc_R1") 
        err2, data2 = robot.robot.GetRobotTeachingPoint("goc_R2") 
        err3, data3 = robot.robot.GetRobotTeachingPoint("goc_R3") 
        err4, data4 = robot.robot.GetRobotTeachingPoint("goc_R4") 

        if any([err1, err2, err3, err4]):
            print(f"Lỗi: Không tìm thấy điểm G1-G4! ({err1}, {err2}, {err3}, {err4})")
            raise Exception("Kiểm tra lại tên biến trong tay Robot!")

        src_pts_robot = np.array([
            [float(data1[0]), float(data1[1])],
            [float(data2[0]), float(data2[1])],
            [float(data3[0]), float(data3[1])],
            [float(data4[0]), float(data4[1])]
        ], dtype=np.float32)

        M_rob = cv2.getPerspectiveTransform(dst_pts_logic, src_pts_robot)
        robot.set_perspective_matrix(M_rob)
        print("=== ROBOT CALIBRATION OK ===")
except Exception as e:
    print(f"[MAIN] Lỗi Calib Robot: {e}"); 
    if not config.DRY_RUN: sys.exit()

# --- LOAD MODEL YOLO ---
MODEL_PATH = r"D:\xiangqi_robot_TrainningAI_Final_4\models_chinesechess1\content\runs\detect\train\weights\best.pt"
try:
    model = YOLO(MODEL_PATH)
except:
    print(f"❌ LỖI: Không tìm thấy file model tại {MODEL_PATH}")
    MODEL_PATH = r"D:\1\xiangqi_robot_TrainningAI_Final_3\models_chinesechess1\content\runs\detect\train\weights\best.pt"
    model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(int(os.environ.get("VIDEO_INDEX", "1")), cv2.CAP_DSHOW)
cap.set(3, 640); cap.set(4, 480)
PERSPECTIVE_PATH = Path(__file__).parent / "perspective.npy"

CLASS_ID_TO_INTERNAL_NAME = {
    0: 'b_A', 1: 'b_C', 2: 'b_R', 3: 'b_E', 4: 'b_K', 5: 'b_N', 6: 'b_P',
    8: 'r_A', 9: 'r_C', 10: 'r_R', 11: 'r_E', 12: 'r_K', 13: 'r_N', 14: 'r_P'
}

# ==========================================
# 4. HÀM HỖ TRỢ
# ==========================================
def grid_to_pixel(col, row):
    return int(START_X + col * SQUARE_SIZE), int(START_Y + row * SQUARE_SIZE)

def pixel_to_grid(px, py):
    col = int(round((px - START_X) / SQUARE_SIZE))
    row = int(round((py - START_Y) / SQUARE_SIZE))
    return col, row

def draw_ui():
    screen.fill(BOARD_COLOR)
    if not game_over:
        pygame.draw.rect(screen, BTN_COLOR, BTN_SURRENDER_RECT, border_radius=8)
        txt = UI_FONT.render("ĐẦU HÀNG", True, (255, 255, 255))
        screen.blit(txt, txt.get_rect(center=BTN_SURRENDER_RECT.center))
        mode_txt = UI_FONT.render(f"MODE: {'MOUSE CLICK' if ALLOW_MOUSE_MOVE else 'CAMERA AI'}", True, (0,0,255))
        screen.blit(mode_txt, (10, 10))
    
    for r in range(NUM_ROWS): pygame.draw.line(screen, LINE_COLOR, grid_to_pixel(0, r), grid_to_pixel(NUM_COLS - 1, r), 1)
    for c in range(NUM_COLS):
        if c in [0, NUM_COLS - 1]: pygame.draw.line(screen, LINE_COLOR, grid_to_pixel(c, 0), grid_to_pixel(c, NUM_ROWS - 1), 1)
        else:
            pygame.draw.line(screen, LINE_COLOR, grid_to_pixel(c, 0), grid_to_pixel(c, 4), 1)
            pygame.draw.line(screen, LINE_COLOR, grid_to_pixel(c, 5), grid_to_pixel(c, 9), 1)
    pygame.draw.line(screen, LINE_COLOR, grid_to_pixel(3, 0), grid_to_pixel(5, 2), 1)
    pygame.draw.line(screen, LINE_COLOR, grid_to_pixel(5, 0), grid_to_pixel(5, 2), 1)
    pygame.draw.line(screen, LINE_COLOR, grid_to_pixel(3, 7), grid_to_pixel(5, 9), 1)
    pygame.draw.line(screen, LINE_COLOR, grid_to_pixel(5, 7), grid_to_pixel(3, 9), 1)

def draw_pieces():
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            name = board[r][c]
            if name == '.': continue
            cx, cy = grid_to_pixel(c, r)
            color = (220, 20, 60) if name.startswith('r') else (0, 0, 0)
            pygame.draw.circle(screen, (255, 255, 255), (cx, cy), PIECE_RADIUS)
            pygame.draw.circle(screen, color, (cx, cy), PIECE_RADIUS, 2)
            text_surf = PIECE_FONT.render(PIECE_DISPLAY_NAMES.get(name, '?'), True, color)
            screen.blit(text_surf, text_surf.get_rect(center=(cx, cy)))

def draw_highlight():
    if last_move:
        (s, d) = last_move
        pygame.draw.circle(screen, (0, 255, 0, 100), grid_to_pixel(s[0], s[1]), PIECE_RADIUS + 2, 2)
        pygame.draw.circle(screen, (0, 255, 0, 150), grid_to_pixel(d[0], d[1]), PIECE_RADIUS + 2, 2)
    if selected_pos:
        c, r = selected_pos
        cx, cy = grid_to_pixel(c, r)
        pygame.draw.circle(screen, (0, 0, 255), (cx, cy), PIECE_RADIUS + 4, 2)

def calibrate_perspective_camera(cap, save_path):
    pts = []
    window = "CALIBRATE"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, lambda e,x,y,f,p: pts.append((x,y)) if e==1 and len(pts)<4 else None)
    print("⚠️ LƯU Ý: Click 4 góc theo thứ tự: 1.TopLeft -> 2.TopRight -> 3.BotRight -> 4.BotLeft")
    while True:
        ret, frame = cap.read()
        if not ret: break
        for i, p in enumerate(pts): 
            cv2.circle(frame, p, 6, (0,255,0), -1)
            cv2.putText(frame, str(i+1), p, 1, 2, (0,255,0))
        cv2.imshow(window, frame)
        if cv2.waitKey(1) == 27 or len(pts)==4: break
    if len(pts)==4:
        dst = np.array([[0, 0], [8, 0], [8, 9], [0, 9]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(np.array(pts, dtype=np.float32), dst)
        np.save(save_path, M); print(f"Đã lưu file hiệu chỉnh: {save_path}")
    cv2.destroyWindow(window)

def detections_to_grid_occupancy(detections, M):
    grid = [['.' for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)]
    if M is None: return grid
    for cls_id, (x1, y1, x2, y2) in detections:
        name = CLASS_ID_TO_INTERNAL_NAME.get(cls_id)
        if not name: continue
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        try:
            dst = cv2.perspectiveTransform(np.array([[[float(cx), float(cy)]]],dtype=np.float32), M)[0][0]
            c, r = int(round(dst[0])), int(round(dst[1]))
            if 0<=c<NUM_COLS and 0<=r<NUM_ROWS: grid[r][c] = name
        except: pass
    return grid

PIECE_LETTER_TO_SOUND = {'N':'Ma', 'C':'Phao', 'R':'Xe', 'P':'Tot', 'A':'Si', 'E':'Tuong', 'K':None}
def piece_str_to_sound(piece_str):
    if not piece_str or piece_str == '.': return None
    try: return PIECE_LETTER_TO_SOUND.get(piece_str.split('_')[-1])
    except: return None

def process_human_move(src, dst, p_name):
    global board, last_move, turn, winner, game_over
    print(f"[HUMAN] ✅ Đã đi: {p_name} {src}->{dst}")
    key = ai_book.board_to_key(board)
    move_history.append({'turn':'r', 'key':key, 'src':src, 'dst':dst})
    cap_p = board[dst[1]][dst[0]]
    if cap_p != '.': b_captured.append(cap_p)
    board, _ = xiangqi.make_temp_move(board, (src, dst))
    last_move = (src, dst)
    attacker_sound = piece_str_to_sound(p_name)
    if cap_p != '.':
        target_sound = piece_str_to_sound(cap_p)
        if attacker_sound and target_sound: sound_player.play_capture_sound(attacker_sound, target_sound)
    else:
        if attacker_sound: sound_player.play_move_sound(attacker_sound)
    if xiangqi.get_king_pos('b', board) is None: 
        winner, game_over = 'r', True
        ai_book.learn_game(move_history, winner)
    else: 
        turn = 'b'

# ==========================================
# 5. VÒNG LẶP CHÍNH
# ==========================================
running = True
last_sync_time = time.time()
SYNC_INTERVAL = 1.5 
clock = pygame.time.Clock()

while running:
    draw_ui(); draw_pieces(); draw_highlight()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_v:
            calibrate_perspective_camera(cap, str(PERSPECTIVE_PATH))
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if BTN_SURRENDER_RECT.collidepoint(mx, my) and not game_over:
                print("[GAME] BẠN ĐẦU HÀNG!"); winner, game_over = 'b', True
                ai_book.learn_game(move_history, winner)

    # --- CAMERA ĐỌC LIÊN TỤC & DEBUG GRID ---
    ret, frame = cap.read()
    detections = []
    if ret:
        try:
            # 1. Detect YOLO
            results = model.predict(frame, conf=0.35, iou=0.45, verbose=False) # Giảm conf xuống 0.50
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls in CLASS_ID_TO_INTERNAL_NAME:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((cls, (x1, y1, x2, y2)))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            
            # 2. VẼ LƯỚI ẢO (DEBUG) ĐỂ KIỂM TRA LỆCH
            if os.path.exists(str(PERSPECTIVE_PATH)):
                M_debug = np.load(str(PERSPECTIVE_PATH))
                M_inv = np.linalg.inv(M_debug)
                for r_debug in range(NUM_ROWS):
                    for c_debug in range(NUM_COLS):
                        pt_grid = np.array([[[float(c_debug), float(r_debug)]]], dtype=np.float32)
                        pt_pixel = cv2.perspectiveTransform(pt_grid, M_inv)[0][0]
                        px, py = int(pt_pixel[0]), int(pt_pixel[1])
                        # Vẽ chấm đỏ
                        cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
                        if c_debug == 0 and r_debug == 0: cv2.putText(frame, "START", (px, py), 1, 1, (0,0,255), 2)

            cv2.imshow("Camera Monitor", frame)
        except: pass
    if cv2.waitKey(1) == ord('q'): running = False

    # --- LOGIC DI CHUYỂN CAMERA ---
    if not ALLOW_MOUSE_MOVE:
        if time.time() - last_sync_time > SYNC_INTERVAL and turn == 'r' and not game_over:
            last_sync_time = time.time()
            M_cam = np.load(str(PERSPECTIVE_PATH)) if os.path.exists(str(PERSPECTIVE_PATH)) else None
            
            if M_cam is not None:
                cam_grid = detections_to_grid_occupancy(detections, M_cam)
                disappeared = []
                appeared = []
                
                for r in range(NUM_ROWS):
                    for c in range(NUM_COLS):
                        if board[r][c] != '.' and (cam_grid[r][c] == '.' or cam_grid[r][c] != board[r][c]):
                            disappeared.append({'pos': (c, r), 'piece': board[r][c]})
                        if cam_grid[r][c] != '.' and cam_grid[r][c] != board[r][c]:
                            appeared.append({'pos': (c, r), 'piece': cam_grid[r][c]})
                
                # In ít log hơn để dễ nhìn
                if disappeared and appeared:
                     print(f"[CAM] Change detected...")

                valid_move = None
                for app in appeared:
                    for dis in disappeared:
                        if app['piece'].startswith('r'):
                            if dis['piece'] == app['piece'] and dis['pos'] != app['pos']:
                                valid_move = (dis['pos'], app['pos'], app['piece'])
                                break
                    if valid_move: break
                
                if valid_move:
                    src, dst, p_name = valid_move
                    if xiangqi.is_valid_move(src, dst, board, 'r'):
                        process_human_move(src, dst, p_name)
                    else:
                        print(f"[IGN] ⚠️ Thấy {src}->{dst} nhưng SAI LUẬT (Check lại Calibration!)")

    # --- AI TURN ---
    if turn == 'b' and not game_over:
        print(f"[AI] Đang nghĩ..."); pygame.display.flip()
        try:
            best = ai.pick_best_move(board, 'b')
            if best:
                try: s, d = best 
                except: s, d = best[0], best[1]
                print(f"[AI] Đi: {s}->{d}")
                
                key = ai_book.board_to_key(board)
                move_history.append({'turn':'b', 'key':key, 'src':s, 'dst':d})

                cap_p = board[d[1]][d[0]]
                is_cap = (cap_p != '.')
                if is_cap: r_captured.append(cap_p); print(f"[AI] Ăn quân: {cap_p}")
                
                if robot.connected:
                    attacker_sound = piece_str_to_sound(board[s[1]][s[0]])
                    if is_cap:
                        target_sound = piece_str_to_sound(cap_p)
                        if attacker_sound and target_sound: sound_player.play_capture_sound(attacker_sound, target_sound)
                    else:
                        if attacker_sound: sound_player.play_move_sound(attacker_sound)
                    robot.move_piece(s[0], s[1], d[0], d[1], is_cap)
                
                board, _ = xiangqi.make_temp_move(board, best)
                last_move = best
                
                if xiangqi.get_king_pos('r', board) is None: 
                    winner, game_over = 'b', True
                    ai_book.learn_game(move_history, winner)
                else: 
                    turn = 'r'
                    print("[GAME] Chờ người chơi..."); last_sync_time = time.time() + 4.0
            else:
                print("[AI] Hết nước"); winner, game_over = 'r', True
                ai_book.learn_game(move_history, winner)
        except Exception as e:
            print(f"AI Error: {e}"); traceback.print_exc(); turn = 'r'

    if game_over:
        txt = GAME_FONT.render(f"{'BẠN' if winner=='r' else 'AI'} THẮNG!", True, (0,255,0))
        screen.blit(txt, txt.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2)))
    
    pygame.display.flip(); clock.tick(30)

cap.release(); cv2.destroyAllWindows()
if robot.connected: robot.robot.RobotEnable(0)
pygame.quit(); sys.exit()