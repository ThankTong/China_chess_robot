# =============================================================================
# === FILE: ai.py (SMART VERSION: BETTER EVAL + STRICT BOOK) ===
# =============================================================================
import time
import math
import xiangqi
import config
import sys
import random 
import importlib

# --- IMPORT BOOK ---
try:
    import ai_book as opening_book
except ImportError:
    print("[AI WARNING] Không tìm thấy 'ai_book.py'.")
    opening_book = None

# --- CẤU HÌNH AI ---
MAX_SEARCH_DEPTH = 6
transposition_table = {} 
KILLER_MOVES = [[None, None] for _ in range(MAX_SEARCH_DEPTH + 1)]
SESSION_LEARNED_MOVES = {} 

# --- ĐIỂM SỐ (Cân chỉnh lại cho chuẩn hơn) ---
# Tăng giá trị Xe/Pháo/Mã lên chút để AI biết giữ quân chủ lực
piece_scores = {
    'r_P': 10, 'r_N': 45, 'r_E': 25, 'r_A': 25, 'r_K': 10000, 'r_C': 55, 'r_R': 110,
    'b_P': -10, 'b_N': -45, 'b_E': -25, 'b_A': -25, 'b_K': -10000, 'b_C': -55, 'b_R': -110,
}

# --- PST TABLES (GIỮ NGUYÊN) ---
PAWN_PST = [[0,3,6,9,12,9,6,3,0],[0,3,6,9,12,9,6,3,0],[0,3,6,9,12,9,6,3,0],[18,36,56,80,120,80,56,36,18],[14,26,42,60,80,60,42,26,14],[10,20,30,34,40,34,30,20,10],[6,12,18,18,20,18,18,12,6],[2,0,8,0,8,0,8,0,2],[0,0,-2,0,4,0,-2,0,0],[0,0,0,0,0,0,0,0,0]]
KNIGHT_PST = [[4,8,12,12,10,12,12,8,4],[4,10,14,16,14,16,14,10,4],[8,12,18,20,18,20,18,12,8],[10,16,22,24,22,24,22,16,10],[8,14,20,22,20,22,20,14,8],[6,12,16,18,16,18,16,12,6],[4,10,14,16,14,16,14,10,4],[4,8,12,14,12,14,12,8,4],[2,4,8,8,6,8,8,4,2],[0,2,4,4,4,4,4,2,0]]
CANNON_PST = [[6,8,8,10,9,10,8,8,6],[6,8,8,9,9,9,8,8,6],[6,6,8,9,9,9,8,6,6],[6,6,6,8,8,8,6,6,6],[6,6,6,6,6,6,6,6,6],[6,6,6,6,6,6,6,6,6],[7,7,7,8,8,8,7,7,7],[9,9,9,10,10,10,9,9,9],[10,10,10,12,12,12,10,10,10],[10,12,12,14,14,14,12,12,10]]
ROOK_PST = [[14,14,16,18,18,18,16,14,14],[16,18,20,24,26,24,20,18,16],[14,16,18,22,24,22,18,16,14],[12,14,16,20,22,20,16,14,12],[12,14,16,20,22,20,16,14,12],[12,12,14,18,20,18,14,12,12],[8,8,10,14,16,14,10,8,8],[4,4,6,10,12,10,6,4,4],[0,0,0,4,6,4,0,0,0],[-2,0,4,6,8,6,4,0,-2]]
ELEPHANT_PST = [[0, 0, 3, 0, 0, 0, 3, 0, 0] if r in [2, 7] else [0] * 9 for r in range(10)]
ADVISOR_PST = [[0] * 9] * 10 
KING_PST = [[0] * 9] * 10

PST_TABLES = {'P': PAWN_PST, 'N': KNIGHT_PST, 'R': ROOK_PST, 'C': CANNON_PST, 'E': ELEPHANT_PST, 'A': ADVISOR_PST, 'K': KING_PST}

# --- HÀM HỖ TRỢ ---
def get_pst_value(piece_type, col, row, color):
    pst = PST_TABLES.get(piece_type)
    if pst is None: return 0
    if color == 'b': return -pst[xiangqi.NUM_ROWS - 1 - row][col]
    else: return pst[row][col]

def get_move_score(board, move):
    (s_col, s_row), (d_col, d_row) = move
    source_piece = board[s_row][s_col]
    target_piece = board[d_row][d_col]
    if source_piece == '.': return 0
    
    score = 0
    # 1. Ưu tiên bắt quân (MVV/LVA)
    if target_piece != '.':
        # Ăn quân càng to càng tốt (Ví dụ Tốt ăn Xe = 10000 + 100 - 10 = 10090)
        score += 10000 + abs(piece_scores.get(target_piece, 0)) - abs(piece_scores.get(source_piece, 0))
    
    # 2. Điểm vị trí
    piece_type = source_piece[2]
    pst = PST_TABLES.get(piece_type)
    if pst:
        color = source_piece[0]
        if color == 'b':
            old = -pst[xiangqi.NUM_ROWS - 1 - s_row][s_col]; new = -pst[xiangqi.NUM_ROWS - 1 - d_row][d_col]
        else:
            old = pst[s_row][s_col]; new = pst[d_row][d_col]
        score += (new - old)
    return score

def store_killer_move(move, depth):
    if depth < 0 or depth >= MAX_SEARCH_DEPTH: return
    if KILLER_MOVES[depth][0] != move:
        KILLER_MOVES[depth][1] = KILLER_MOVES[depth][0]
        KILLER_MOVES[depth][0] = move

# --- HÀM ĐÁNH GIÁ (NÂNG CẤP: THÊM MOBILITY) ---
def evaluate_board(board):
    score = 0
    mat_diff = 0 
    r_king = None; b_king = None
    
    # Các trọng số thưởng/phạt (Tùy chỉnh để AI có cá tính)
    BONUS_CROSS_RIVER = 20  # Thưởng khi sang sông áp sát
    BONUS_ROOK_OPEN = 15    # Thưởng cho Xe ở cột thoáng
    BONUS_CANNON_HOLLOW = 15 # Thưởng cho Pháo trống (Pháo đầu)
    
    # Duyệt bàn cờ
    for r in range(10):
        for c in range(9):
            p = board[r][c]
            if p == '.': continue
            
            color = p[0]
            type_ = p[2]
            val = piece_scores.get(p, 0)
            pst_val = get_pst_value(type_, c, r, color)
            
            # --- 1. ĐIỂM CƠ BẢN (Vật chất + Vị trí) ---
            current_score = val + pst_val
            
            # --- 2. CHIẾN THUẬT & MOBILITY GIẢ LẬP ---
            
            # [QUÂN TỐT]: Sang sông được thưởng thêm
            if type_ == 'P':
                if color == 'r' and r <= 4: current_score += BONUS_CROSS_RIVER
                elif color == 'b' and r >= 5: current_score -= BONUS_CROSS_RIVER # (Quân đen là điểm âm)

            # [QUÂN XE]: Thưởng nếu ở cột thoáng (không có Tốt cùng màu chặn)
            # [QUÂN XE]: Thưởng nếu ở cột thoáng (không có Tốt cùng màu chặn)
            elif type_ == 'R':
                # Logic: Kiểm tra ô ngay phía trước mặt Xe có trống không?
                # Nếu trống -> Xe cơ động -> Cộng điểm
                
                if color == 'r': # Xe Đỏ (đi từ dưới lên -> hàng giảm dần)
                    # Nếu chưa kịch trần và ô phía trước là khoảng trống
                    if r > 0 and board[r-1][c] == '.':
                        current_score += BONUS_ROOK_OPEN
                else: # Xe Đen (đi từ trên xuống -> hàng tăng dần)
                    # Nếu chưa kịch đáy và ô phía trước là khoảng trống
                    if r < 9 and board[r+1][c] == '.':
                        current_score -= BONUS_ROOK_OPEN # (Trừ điểm vì Đen càng âm càng tốt)

            # [QUÂN PHÁO]: Thích nấp sau quân (Ngòi) hoặc chiếm trung tâm
            elif type_ == 'C':
                if c == 4: # Chiếm trung lộ (Pháo đầu)
                     if color == 'r': current_score += BONUS_CANNON_HOLLOW
                     else: current_score -= BONUS_CANNON_HOLLOW

            # [QUÂN MÃ]: Sợ bị cản chân (Logic này khó viết nhanh, tạm bỏ qua để giữ tốc độ)

            # Cộng dồn vào tổng điểm
            score += current_score
            mat_diff += val
            
            # Lưu vị trí Tướng để tính Tàn cuộc
            if type_ == 'K':
                if color == 'r': r_king = (r, c)
                else: b_king = (r, c)

    # --- 3. LOGIC TÀN CUỘC (Dồn Vua) ---
    # Nếu phe nào hơn quân nhiều (>150 điểm ~ hơn 1 Pháo/Mã), sẽ tìm cách dồn Vua đối phương
    if abs(mat_diff) > 150:
        if mat_diff > 0 and b_king: # Đỏ đang thắng thế
            # Ưu tiên dồn Vua đen vào góc hoặc ép lên cao
            # (Càng lệch tâm 4, càng lệch trục dọc 4 là bị dồn)
            dist = abs(1 - b_king[0]) + abs(4 - b_king[1])
            score += dist * 10 
        elif mat_diff < 0 and r_king: # Đen đang thắng thế
            dist = abs(8 - r_king[0]) + abs(4 - r_king[1])
            score -= dist * 10 # Điểm âm càng nhỏ càng tốt cho Đen

    return score
# --- QUIESCENCE SEARCH ---
def quiescence_search(board, depth, alpha, beta, is_maximizing):
    static_eval = evaluate_board(board)
    if depth == 0: return static_eval

    if is_maximizing:
        if static_eval >= beta: return beta
        alpha = max(alpha, static_eval)
        moves = xiangqi.find_all_valid_moves('r', board)
        caps = [m for m in moves if board[m[1][1]][m[1][0]] != '.']
        caps.sort(key=lambda m: get_move_score(board, m), reverse=True)
        for m in caps:
            nb, _ = xiangqi.make_temp_move(board, m) 
            score = quiescence_search(nb, depth - 1, alpha, beta, False)
            alpha = max(alpha, score)
            if alpha >= beta: break
        return alpha
    else:
        if static_eval <= alpha: return alpha
        beta = min(beta, static_eval)
        moves = xiangqi.find_all_valid_moves('b', board)
        caps = [m for m in moves if board[m[1][1]][m[1][0]] != '.']
        caps.sort(key=lambda m: get_move_score(board, m), reverse=True)
        for m in caps:
            nb, _ = xiangqi.make_temp_move(board, m)
            score = quiescence_search(nb, depth - 1, alpha, beta, True)
            beta = min(beta, score)
            if alpha >= beta: break
        return beta

# --- MINIMAX ---
def minimax(board, depth, is_maximizing, alpha, beta, current_hash, used_depth):
    # Tra bảng băm
    if (current_hash, depth, is_maximizing) in transposition_table:
        return transposition_table[(current_hash, depth, is_maximizing)]

    if depth == 0: return quiescence_search(board, 2, alpha, beta, is_maximizing) # QS depth=2

    color = 'r' if is_maximizing else 'b'
    moves = xiangqi.find_all_valid_moves(color, board)
    
    caps = []; quiets = []
    for m in moves:
        if board[m[1][1]][m[1][0]] != '.': caps.append((get_move_score(board, m), m))
        else: quiets.append(m)
    caps.sort(key=lambda x: x[0], reverse=True)
    ordered = [m for s, m in caps]
    
    ply = used_depth - depth
    if ply <= MAX_SEARCH_DEPTH:
        for k in KILLER_MOVES[ply]:
            if k and k in quiets: quiets.remove(k); ordered.append(k)
    ordered.extend(quiets)

    if not ordered:
        return -math.inf if is_maximizing else math.inf if xiangqi.is_king_in_check(color, board) else 0

    if is_maximizing:
        max_eval = -math.inf
        for m in ordered:
            nb, nh = xiangqi.make_temp_move(board, m, current_hash)
            eval_ = minimax(nb, depth - 1, False, alpha, beta, nh, used_depth)
            max_eval = max(max_eval, eval_)
            alpha = max(alpha, max_eval)
            if beta <= alpha:
                if board[m[1][1]][m[1][0]] == '.': store_killer_move(m, ply)
                break
        transposition_table[(current_hash, depth, is_maximizing)] = max_eval
        return max_eval
    else:
        min_eval = math.inf
        for m in ordered:
            nb, nh = xiangqi.make_temp_move(board, m, current_hash)
            eval_ = minimax(nb, depth - 1, True, alpha, beta, nh, used_depth)
            min_eval = min(min_eval, eval_)
            beta = min(beta, min_eval)
            if beta <= alpha:
                if board[m[1][1]][m[1][0]] == '.': store_killer_move(m, ply)
                break
        transposition_table[(current_hash, depth, is_maximizing)] = min_eval
        return min_eval

# --- PICK BEST MOVE (SIẾT CHẶT LOGIC) ---
def pick_best_move(board_logic, player_color, depth=None):
    # KHÔNG reset bảng băm mỗi nước đi nữa!
    # Chỉ reset nếu nó quá lớn để giải phóng RAM (ví dụ > 1 triệu thế cờ)
    global transposition_table
    if len(transposition_table) > 1000000: 
        transposition_table.clear()
        print("[AI] 🧹 Dọn dẹp bộ nhớ đệm...")
    
    
    # ... (phần còn lại giữ nguyên)
    current_hash = xiangqi.get_zobrist_key(board_logic)
    
    # 1. BOOK: Chỉ dùng nếu nước đi KHÔNG QUÁ TỆ
    if player_color == 'b' and opening_book:
        book_move = opening_book.get_book_move(current_hash)
        if book_move:
            # Thẩm định nhanh: Đi thử xem điểm số thế nào
            temp_board, _ = xiangqi.make_temp_move(board_logic, book_move)
            score_check = evaluate_board(temp_board)
            
            
            # AI là Đen (Minimize). Điểm càng thấp càng tốt.
            # Nếu điểm > 50 (tức là Đỏ đang lời nửa con chốt hoặc thế trận cân bằng) -> OK
            # Nhưng nếu điểm > 150 (Đỏ lời to) -> Book ngu -> Bỏ qua.
            # Lưu ý: Hàm evaluate trả về điểm nhìn từ phe Đỏ (+ cho Đỏ, - cho Đen)
            
            if score_check < 50: # Ngưỡng khắt khe hơn (Cũ là 200)
                print(f"[AI] 📚 Book OK (Eval {score_check}).")
                time.sleep(random.uniform(0.3, 0.8))
                return book_move
            else:
                print(f"[AI] ⚠️ Book TỆ (Eval {score_check}). Bỏ sách, tự tính!")

    # 2. TIME MANAGEMENT
    time_lim = config.AI_THINK_TIME
    max_d = 20
    min_safe = 4 # Bắt buộc sâu ít nhất 2
    
    if depth: 
        time_lim = 99999; max_d = depth; min_safe = depth
        print(f"[AI] ⚙️ Training Depth: {depth}")
    else:
        # Auto depth base on pieces
        pieces = sum(1 for r in range(10) for c in range(9) if board_logic[r][c] != '.')
        print(f"[AI] ⏳ Suy nghĩ (Max {time_lim}s, {pieces} quân)...")

    start = time.time()
    moves = xiangqi.find_all_valid_moves(player_color, board_logic)
    if not moves: return None
    
    # Sắp xếp ưu tiên ăn quân
    moves.sort(key=lambda m: get_move_score(board_logic, m), reverse=True)
    
    best_mv = moves[0] # Nước đi an toàn mặc định

    for d in range(1, max_d + 1):
        # Kiểm tra thời gian
        if time.time() - start > time_lim and d > min_safe:
            print(f"[AI] ⏰ Hết giờ (Xong Depth {d-1})"); break
        
        # Đưa nước tốt nhất lượt trước lên đầu (Move Ordering)
        if best_mv in moves:
            moves.remove(best_mv)
            moves.insert(0, best_mv)
        
        best_sc = -math.inf if player_color == 'r' else math.inf
        cur_best = None
        alpha, beta = -math.inf, math.inf
        
        try:
            for m in moves:
                # Check timeout ngay trong vòng lặp
                if time.time() - start > time_lim * 1.2 and d > min_safe: raise TimeoutError
                
                nb, nh = xiangqi.make_temp_move(board_logic, m, current_hash)
                
                # Gọi Minimax
                sc = minimax(nb, d - 1, player_color == 'r', alpha, beta, nh, d)
                
                if player_color == 'r':
                    if sc > best_sc: best_sc = sc; cur_best = m
                    alpha = max(alpha, sc)
                else:
                    if sc < best_sc: best_sc = sc; cur_best = m
                    beta = min(beta, sc)
            
            if cur_best: best_mv = cur_best
            
            # In tiến độ để bạn thấy nó khôn lên
            # print(f"   -> Depth {d}: Score {best_sc} Move {best_mv}")
            
        except TimeoutError: break
    
    print(f"[AI] ✅ Chốt: {best_mv} (Time {time.time()-start:.2f}s)")
    
    # Lưu học tập (chỉ lưu nếu AI thắng cuối ván)
    if best_mv and player_color == 'b' and opening_book:
        SESSION_LEARNED_MOVES[current_hash] = [best_mv]

    return best_mv

def save_learned_moves(): pass