import pygame
import random
import os

# Khởi tạo pygame mixer
pygame.mixer.init()

# Thư mục gốc chứa các file âm thanh
BASE_SOUND_PATH = "sounds"

# Tên quân chuẩn hoá (để tránh sai chính tả)
VALID_PIECES = ["Ma", "Phao", "Xe", "Tot", "Si", "Tuong"]

# Tạo mapping âm thanh dựa trên filesystem: chỉ tạo entry khi folder thực sự tồn tại
SOUND_MAP = {}

# Cấu trúc repo đang dùng: sounds/Voice <Attacker>/<Attacker> bat <Target>/...
try:
    for attacker_dir in os.listdir(BASE_SOUND_PATH):
        attacker_dir_path = os.path.join(BASE_SOUND_PATH, attacker_dir)
        if not os.path.isdir(attacker_dir_path):
            continue

        # Lấy tên quân từ tên thư mục cuối (ví dụ: 'Voice Ma' -> 'Ma')
        attacker_name = attacker_dir.split()[-1].capitalize()
        if attacker_name not in VALID_PIECES:
            continue

        for sub in os.listdir(attacker_dir_path):
            sub_path = os.path.join(attacker_dir_path, sub)
            if not os.path.isdir(sub_path):
                continue

            # sub ví dụ: 'Ma bat Phao' -> target = 'Phao'
            tokens = sub.split()
            if len(tokens) >= 3 and tokens[-2].lower() == 'bat':
                target_name = tokens[-1].capitalize()
            else:
                continue

            if target_name not in VALID_PIECES:
                continue

            # Lưu đường dẫn tương đối bên trong folder `sounds` (để dùng với os.path.join(BASE_SOUND_PATH, ...))
            SOUND_MAP[(attacker_name, target_name)] = os.path.join(attacker_dir, sub)
except FileNotFoundError:
    # Nếu folder `sounds` không tồn tại, để SOUND_MAP rỗng và in cảnh báo khi cần
    SOUND_MAP = {}


def play_random_sound(subfolder):
    """
    subfolder: đường dẫn trong thư mục sounds.
               VD: "Voice Ma/Ma bat Phao"
    """
    folder_path = os.path.join(BASE_SOUND_PATH, subfolder)

    if not os.path.isdir(folder_path):
        print("⚠ Không tìm thấy folder âm thanh:", folder_path)
        return False

    files = [f for f in os.listdir(folder_path) if f.endswith(".mp3")]
    if not files:
        print("⚠ Không có file mp3 trong:", folder_path)
        return False

    chosen = random.choice(files)
    full_path = os.path.join(folder_path, chosen)

    print(f"🔊 Phát âm thanh: {full_path}")

    try:
        pygame.mixer.music.load(full_path)
        pygame.mixer.music.play()
        return True
    except Exception as e:
        print("❌ Lỗi phát âm:", e)
        return False


def play_capture_sound(attacker, target):
    """
    attacker: quân đi (Ma, Xe, Phao…)
    target: quân bị ăn
    """

    # Chuẩn hoá input
    attacker = attacker.capitalize()
    target = target.capitalize()

    if attacker not in VALID_PIECES:
        print("⚠ Quân tấn công không hợp lệ:", attacker)
        return
    
    if target not in VALID_PIECES:
        print("⚠ Quân bị ăn không hợp lệ:", target)
        return

    folder = SOUND_MAP.get((attacker, target))

    if not folder:
        print(f"⚠ Không có mapping âm thanh cho {attacker} ăn {target}")
        return
    
    play_random_sound(folder)


def play_move_sound(piece):
    """
    Âm thanh di chuyển bình thường (không ăn quân)
    Bro có thể làm sẵn folder:
    sounds/Move/Ma/
    sounds/Move/Xe/
    ...
    """

    piece = piece.capitalize()

    folder = f"Move/{piece}"

    play_random_sound(folder)
