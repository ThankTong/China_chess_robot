# =============================================================================
# === FILE: xiangqi.py (CẬP NHẬT TƯƠNG THÍCH AI) ===
# =============================================================================
import random

NUM_COLS = 9
NUM_ROWS = 10

# --- ZOBRIST HASHING INIT ---
# (Khởi tạo bảng số ngẫu nhiên để tính Key Hash cho AI)
random.seed(2024)
_PIECE_TO_INT = {
    'b_K': 0, 'b_A': 1, 'b_E': 2, 'b_N': 3, 'b_R': 4, 'b_C': 5, 'b_P': 6,
    'r_K': 7, 'r_A': 8, 'r_E': 9, 'r_N': 10, 'r_R': 11, 'r_C': 12, 'r_P': 13
}
_ZOBRIST_TABLE = []
for _r in range(10):
    _row_data = []
    for _c in range(9):
        _pieces_random = [random.getrandbits(64) for _ in range(14)]
        _row_data.append(_pieces_random)
    _ZOBRIST_TABLE.append(_row_data)

initial_board = [
    ['b_R', 'b_N', 'b_E', 'b_A', 'b_K', 'b_A', 'b_E', 'b_N', 'b_R'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', 'b_C', '.', '.', '.', '.', '.', 'b_C', '.'],
    ['b_P', '.', 'b_P', '.', 'b_P', '.', 'b_P', '.', 'b_P'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['r_P', '.', 'r_P', '.', 'r_P', '.', 'r_P', '.', 'r_P'],
    ['.', 'r_C', '.', '.', '.', '.', '.', 'r_C', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['r_R', 'r_N', 'r_E', 'r_A', 'r_K', 'r_A', 'r_E', 'r_N', 'r_R']
]

def get_board():
    return [row[:] for row in initial_board]

def get_zobrist_key(board):
    """Tính Hash cho toàn bộ bàn cờ"""
    h = 0
    for r in range(10):
        for c in range(9):
            piece = board[r][c]
            if piece != '.':
                p_idx = _PIECE_TO_INT.get(piece)
                if p_idx is not None:
                    h ^= _ZOBRIST_TABLE[r][c][p_idx]
    return h

# Hàm này để tương thích với main.py cũ (nếu có gọi)
def get_board_key(board):
    return str(get_zobrist_key(board))

# === HÀM DI CHUYỂN QUAN TRỌNG (ĐÃ SỬA LỖI) ===
def make_temp_move(board, move, current_hash=None):
    """
    Thực hiện nước đi giả lập.
    Nhận vào: board, move, hash hiện tại (tùy chọn).
    Trả về: (Bàn cờ mới, Hash mới).
    """
    new_board = [row[:] for row in board]
    (s_col, s_row), (d_col, d_row) = move
    
    piece = new_board[s_row][s_col]
    captured = new_board[d_row][d_col]
    
    new_hash = current_hash
    
    # Cập nhật Hash nhanh (nếu có hash cũ)
    if new_hash is not None:
        # 1. Xóa quân bị ăn (nếu có)
        if captured != '.':
            cap_idx = _PIECE_TO_INT.get(captured)
            if cap_idx is not None:
                new_hash ^= _ZOBRIST_TABLE[d_row][d_col][cap_idx]
        
        # 2. Xóa quân ở vị trí cũ
        p_idx = _PIECE_TO_INT.get(piece)
        if p_idx is not None:
            new_hash ^= _ZOBRIST_TABLE[s_row][s_col][p_idx]
            # 3. Thêm quân vào vị trí mới
            new_hash ^= _ZOBRIST_TABLE[d_row][d_col][p_idx]
    
    # Thực hiện di chuyển logic
    new_board[d_row][d_col] = piece
    new_board[s_row][s_col] = '.'
    
    # Nếu không có hash đầu vào, tính lại từ đầu
    if new_hash is None:
        new_hash = get_zobrist_key(new_board)
        
    return new_board, new_hash
# ===============================================

def count_pieces_between(source_coords, dest_coords, board):
    (s_col, s_row) = source_coords
    (d_col, d_row) = dest_coords
    count = 0
    if s_col == d_col:
        for r in range(min(s_row, d_row) + 1, max(s_row, d_row)):
            if board[r][s_col] != '.': count += 1
    elif s_row == d_row:
        for c in range(min(s_col, d_col) + 1, max(s_col, d_col)):
            if board[s_row][c] != '.': count += 1
    return count

def get_king_pos(color, board):
    king_char = color + '_K'
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            if board[r][c] == king_char:
                return (c, r)
    return None

def is_across_river(row, color_prefix):
    if color_prefix == 'r': return row <= 4
    else: return row >= 5

def is_in_palace(col, row, color_prefix):
    if col < 3 or col > 5: return False
    if color_prefix == 'r': return row >= 7
    else: return row <= 2

def is_kings_facing(board):
    r_king = get_king_pos('r', board)
    b_king = get_king_pos('b', board)
    if r_king is None or b_king is None: return False
    if r_king[0] != b_king[0]: return False
    return count_pieces_between(r_king, b_king, board) == 0

# --- LOGIC LUẬT DI CHUYỂN ---
def _is_valid_pawn_move(s, d, c, board):
    (sc, sr), (dc, dr) = s, d
    if c == 'r':
        if dr == sr - 1 and dc == sc: return True
        if dr == sr and abs(dc - sc) == 1 and is_across_river(sr, 'r'): return True
    else:
        if dr == sr + 1 and dc == sc: return True
        if dr == sr and abs(dc - sc) == 1 and is_across_river(sr, 'b'): return True
    return False

def _is_valid_rook_move(s, d, board):
    if s[0] != d[0] and s[1] != d[1]: return False
    return count_pieces_between(s, d, board) == 0

def _is_valid_knight_move(s, d, board):
    dc, dr = abs(s[0] - d[0]), abs(s[1] - d[1])
    if (dc, dr) not in [(1, 2), (2, 1)]: return False
    if dc == 2: return board[s[1]][(s[0] + d[0])//2] == '.'
    else: return board[(s[1] + d[1])//2][s[0]] == '.'

def _is_valid_cannon_move(s, d, board):
    if s[0] != d[0] and s[1] != d[1]: return False
    obs = count_pieces_between(s, d, board)
    if board[d[1]][d[0]] == '.': return obs == 0
    return obs == 1

def _is_valid_elephant_move(s, d, c, board):
    if abs(s[0] - d[0]) != 2 or abs(s[1] - d[1]) != 2: return False
    if is_across_river(d[1], c): return False
    if board[(s[1] + d[1])//2][(s[0] + d[0])//2] != '.': return False
    return True

def _is_valid_advisor_move(s, d, c):
    if abs(s[0] - d[0]) == 1 and abs(s[1] - d[1]) == 1:
        return is_in_palace(d[0], d[1], c)
    return False

def _is_valid_king_move(s, d, c):
    if not is_in_palace(d[0], d[1], c): return False
    return abs(s[0] - d[0]) + abs(s[1] - d[1]) == 1

def is_valid_move(source, dest, board, color):
    s_col, s_row = source
    d_col, d_row = dest
    piece = board[s_row][s_col]
    if piece == '.' or piece[0] != color: return False
    dest_p = board[d_row][d_col]
    if dest_p != '.' and dest_p[0] == color: return False
    
    type_ = piece[2]
    valid = False
    if type_ == 'P': valid = _is_valid_pawn_move(source, dest, color, board)
    elif type_ == 'R': valid = _is_valid_rook_move(source, dest, board)
    elif type_ == 'N': valid = _is_valid_knight_move(source, dest, board)
    elif type_ == 'C': valid = _is_valid_cannon_move(source, dest, board)
    elif type_ == 'E': valid = _is_valid_elephant_move(source, dest, color, board)
    elif type_ == 'A': valid = _is_valid_advisor_move(source, dest, color)
    elif type_ == 'K': valid = _is_valid_king_move(source, dest, color)
    
    if not valid: return False
    
    # Check lộ mặt tướng
    tmp, _ = make_temp_move(board, (source, dest))
    if is_kings_facing(tmp): return False
    if is_king_in_check(color, tmp): return False
    
    return True

def find_all_valid_moves(color, board):
    moves = []
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            if board[r][c] != '.' and board[r][c][0] == color:
                for rr in range(NUM_ROWS):
                    for cc in range(NUM_COLS):
                        if is_valid_move((c,r), (cc,rr), board, color):
                            moves.append(((c,r), (cc,rr)))
    return moves

def is_king_in_check(color, board):
    k_pos = get_king_pos(color, board)
    if not k_pos: return False
    opp = 'b' if color == 'r' else 'r'
    
    # Kiểm tra bị quân nào tấn công không
    # (Viết gọn để tránh đệ quy sâu)
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            p = board[r][c]
            if p != '.' and p[0] == opp:
                # Kiểm tra nhanh, bỏ qua self-check (vì đang check xem Vua có bị ăn không)
                t = p[2]
                v = False
                s, d = (c,r), k_pos
                if t == 'P': v = _is_valid_pawn_move(s, d, opp, board)
                elif t == 'R': v = _is_valid_rook_move(s, d, board)
                elif t == 'N': v = _is_valid_knight_move(s, d, board)
                elif t == 'C': v = _is_valid_cannon_move(s, d, board)
                elif t == 'K': v = (s[0] == d[0] and count_pieces_between(s, d, board) == 0)
                if v: return True
    return False