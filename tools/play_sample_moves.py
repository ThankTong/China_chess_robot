import time
import sys
import os

# Ensure project root is on sys.path so we can import sound_player from tools/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

# Import sound_player (khởi tạo pygame mixer khi import)
import sound_player

print('Testing sound playback...')

# Test move sound (non-capture)
print('Play move sound: Ma')
sound_player.play_move_sound('Ma')
# chờ để nghe
time.sleep(2)

# Test capture sound (existing)
print('Play capture sound: Ma eats Phao')
sound_player.play_capture_sound('Ma', 'Phao')
# chờ để nghe
time.sleep(3)

# Test capture sound for a missing mapping (should print warning)
print('Play capture sound: Si eats Si (expected missing)')
sound_player.play_capture_sound('Si', 'Si')

print('Done.')
