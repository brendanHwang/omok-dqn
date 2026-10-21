"""Gomocup 기보에서 같은 플레이어의 두 차례 사이 전이를 구성한다."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np


def has_five(board, x, y, player):
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            xx, yy = x + sign * dx, y + sign * dy
            while 0 <= xx < 20 and 0 <= yy < 20 and board[yy, xx] == player:
                count += 1
                xx += sign * dx
                yy += sign * dy
        if count >= 5:
            return True
    return False


def parse_game(text):
    lines = text.splitlines()
    if not lines or '20x20' not in lines[0]:
        raise ValueError('20x20 기보가 아닙니다')
    board = np.zeros((20, 20), dtype=np.int8)
    states, actions = [], []
    winner = 0
    for line in lines[1:]:
        if not re.fullmatch(r'\d+,\d+,\d+', line):
            break
        x, y, _ = map(int, line.split(','))
        x -= 1
        y -= 1
        if not (0 <= x < 20 and 0 <= y < 20) or board[y, x] != 0:
            raise ValueError('범위 밖 또는 중복 착수입니다')
        if winner:
            raise ValueError('승패가 결정된 이후 착수가 있습니다')
        player = 1 + len(actions) % 2
        states.append(board.copy())
        actions.append(y * 20 + x)
        board[y, x] = player
        if has_five(board, x, y, player):
            winner = player
    if not winner and np.any(board == 0):
        raise ValueError('종료 결과를 판정할 수 없는 기보입니다')
    states.append(board.copy())
    return states, actions, winner


def transitions(states, actions, winner):
    result = []
    for i, action in enumerate(actions):
        player = 1 + i % 2
        # 상대 착수까지 포함해야 다음 상태도 같은 플레이어의 차례가 된다.
        end = min(i + 2, len(actions))
        done = end == len(actions)
        reward = (1 if winner == player else -1) if done and winner else 0
        encode = lambda b: np.where(b == player, 1, np.where(b == 0, 0, -1)).astype(np.int8)
        result.append((encode(states[i]), action, reward, encode(states[end]), done))
    return result


def build_dataset(archive, output):
    groups = {'train': [], 'validation': []}
    seen, names = set(), {'train': [], 'validation': []}
    stats = {'source_games': 0, 'excluded_games': 0, 'duplicate_games': 0}
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if 'Freestyle' not in name or not name.endswith('.psq'):
                continue
            stats['source_games'] += 1
            try:
                states, actions, winner = parse_game(z.read(name).decode(errors='replace'))
            except ValueError:
                stats['excluded_games'] += 1
                continue
            digest = hashlib.sha256(np.asarray(actions, dtype=np.int16).tobytes()).hexdigest()
            if digest in seen:
                stats['duplicate_games'] += 1
                continue
            seen.add(digest)
            # 완전히 동일한 기보를 제거하고 대국 단위로 약 90:10 분리한다.
            split = 'validation' if int(digest[:8], 16) % 10 == 0 else 'train'
            groups[split].extend(transitions(states, actions, winner))
            names[split].append(name)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in groups.items():
        s, a, r, ns, d = zip(*rows)
        np.savez_compressed(output / f'{split}.npz', states=np.stack(s),
                            actions=np.asarray(a, dtype=np.int64), rewards=np.asarray(r, dtype=np.float32),
                            next_states=np.stack(ns), dones=np.asarray(d, dtype=bool))
        stats[f'{split}_games'] = len(names[split])
        stats[f'{split}_transitions'] = len(rows)
    stats['game_files'] = names
    stats['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps({k:v for k,v in stats.items() if k != 'game_files'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('dataset/dqn'))
    args = parser.parse_args()
    build_dataset(args.archive, args.output)
