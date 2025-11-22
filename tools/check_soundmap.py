import os, json

BASE_SOUND_PATH = "sounds"
VALID_PIECES = ["Ma", "Phao", "Xe", "Tot", "Si", "Tuong"]

sound_map = {}

if not os.path.isdir(BASE_SOUND_PATH):
    print("⚠ Không tìm thấy thư mục sounds, đảm bảo đang ở đúng thư mục dự án.")
    raise SystemExit(1)

for attacker_dir in os.listdir(BASE_SOUND_PATH):
    attacker_dir_path = os.path.join(BASE_SOUND_PATH, attacker_dir)
    if not os.path.isdir(attacker_dir_path):
        continue

    attacker_name = attacker_dir.split()[-1].capitalize()
    if attacker_name not in VALID_PIECES:
        continue

    for sub in os.listdir(attacker_dir_path):
        sub_path = os.path.join(attacker_dir_path, sub)
        if not os.path.isdir(sub_path):
            continue

        tokens = sub.split()
        if len(tokens) >= 3 and tokens[-2].lower() == 'bat':
            target_name = tokens[-1].capitalize()
        else:
            continue

        if target_name not in VALID_PIECES:
            continue

        sound_map[(attacker_name, target_name)] = os.path.join(attacker_dir, sub)

pairs = [f"{a}->{b}: {sound_map[(a,b)]}" for (a,b) in sorted(sound_map.keys())]
print(json.dumps(pairs, ensure_ascii=False, indent=2))

missing = []
for a in VALID_PIECES:
    for b in VALID_PIECES:
        if (a,b) not in sound_map:
            missing.append(f"{a}->{b}")

print('\nMissing count:', len(missing))
print(json.dumps(sorted(missing), ensure_ascii=False, indent=2))

print('\nCheck sample folders:')
for idx, ((a,b), v) in enumerate(sorted(sound_map.items())):
    if idx >= 20:
        break
    path = os.path.join(BASE_SOUND_PATH, v)
    print(f"{a}->{b} -> {path} exists: {os.path.isdir(path)}")

print('\nDone.')
