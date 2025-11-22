import json
import os
import random
import xiangqi

# Tên file để lưu các nước đi mới học được
NEW_LEARNED_FILE = "learned_data.json"

# --- DỮ LIỆU BOOK CỦA BẠN (HASH KEY) ---
# (Tôi để mẫu 2 dòng, bạn hãy DÁN TOÀN BỘ dữ liệu BOOK khổng lồ của bạn vào đây)
BOOK = {
   
}

# --- BIẾN TOÀN CỤC ĐỂ LƯU RAM (CACHE) ---
LEARNED_CACHE = {}

def board_to_key(board):
    if isinstance(board, str) or isinstance(board, int): return str(board)
    return str(xiangqi.get_zobrist_key(board))

def load_learned_data_to_memory():
    """Chỉ gọi hàm này 1 lần khi khởi động"""
    global LEARNED_CACHE
    if not os.path.exists(NEW_LEARNED_FILE): 
        LEARNED_CACHE = {}
        return
    try:
        with open(NEW_LEARNED_FILE, "r", encoding="utf-8") as f:
            LEARNED_CACHE = json.load(f)
        print(f"[BOOK] Đã nạp {len(LEARNED_CACHE)} thế cờ đã học vào RAM.")
    except: LEARNED_CACHE = {}

def save_book_to_disk():
    """Lưu RAM xuống đĩa"""
    try:
        with open(NEW_LEARNED_FILE, "w", encoding="utf-8") as f:
            json.dump(LEARNED_CACHE, f, indent=None)
    except Exception as e: print(f"[BOOK] Lỗi lưu file: {e}")

# --- GỌI NẠP DỮ LIỆU NGAY KHI IMPORT ---
load_learned_data_to_memory()

def get_book_move(input_data):
    key = board_to_key(input_data)
    move_list = None

    # 1. Ưu tiên Book gốc
    if key in BOOK:
        move_list = BOOK[key]
    # 2. Tìm trong RAM (Nhanh gấp 1000 lần đọc đĩa)
    elif key in LEARNED_CACHE:
        move_list = LEARNED_CACHE[key]

    if move_list:
        choice = random.choice(move_list)
        try:
            src = (choice[0][0], choice[0][1])
            dst = (choice[1][0], choice[1][1])
            return (src, dst)
        except: return None
    return None

def learn_game(move_history, winner):
    """Học và lưu vào RAM + Đĩa"""
    print(f"[BOOK] Đang học từ ván đấu... (Winner: {winner})")
    if not move_history: return

    global LEARNED_CACHE
    count = 0
    
    for rec in move_history:
        if rec['turn'] == winner:
            key = str(rec.get('key') or rec.get('board_key'))
            if not key: continue
            
            move = [list(rec['src']), list(rec['dst'])]
            
            if key not in LEARNED_CACHE:
                LEARNED_CACHE[key] = [move]
                count += 1
            elif move not in LEARNED_CACHE[key]:
                LEARNED_CACHE[key].append(move)
                count += 1
    
    if count > 0:
        save_book_to_disk() # Chỉ lưu xuống đĩa khi có cái mới
        print(f"[BOOK] Đã học thêm {count} nước đi mới!")
    else:
        print("[BOOK] Không có gì mới để học.")