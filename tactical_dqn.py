"""DQN 후보 점수와 승리/방어 전술을 결합한 4수 및 강제 방어 탐색 에이전트."""
import numpy as np


class TacticalDQN:
    def __init__(self, width=12, replies=8, q_weight=.03, depth=4):
        self.depth = depth
        self.width = width
        self.replies = replies
        self.q_weight = q_weight
        # 모든 가로·세로·대각선의 연속 다섯 교점을 한 번만 구성한다.
        lines = []
        for y in range(20):
            for x in range(20):
                for dx, dy in ((1,0),(0,1),(1,1),(1,-1)):
                    if 0 <= x+4*dx < 20 and 0 <= y+4*dy < 20:
                        lines.append([(y+i*dy)*20+x+i*dx for i in range(5)])
        self.lines = np.asarray(lines)
        self.weights = np.asarray([0., 1., 12., 180., 16000., 1.e8])

    def analyse(self, board, player, defense=True):
        flat = board.ravel()
        cells = flat[self.lines]
        own = (cells == player).sum(1)
        enemy = (cells == 3-player).sum(1)
        empty = cells == 0
        own_weight = np.where(enemy == 0, self.weights[own], 0.)
        enemy_weight = np.where(own == 0, self.weights[enemy], 0.)
        scores = np.bincount(self.lines.ravel(),
                             weights=((own_weight + (1.12*enemy_weight if defense else 0))[:,None]*empty).ravel(),
                             minlength=400)
        win_lines = (own == 4) & (enemy == 0)
        winning = np.unique(self.lines[win_lines][empty[win_lines]])
        scores[flat != 0] = -np.inf
        return scores, winning

    def candidates(self, board, player, count):
        scores, wins = self.analyse(board, player)
        if len(wins):
            return wins, scores
        _, blocks = self.analyse(board, 3-player)
        if len(blocks):
            return blocks, scores
        # 위협 점수가 높은 후보부터 탐색 폭만큼 조사한다.
        valid = np.flatnonzero(board.ravel() == 0)
        order = valid[np.argsort(-scores[valid], kind='stable')[:count]]
        return order, scores

    def value(self, board, player):
        own, _ = self.analyse(board, player, defense=False)
        enemy, _ = self.analyse(board, 3-player, defense=False)
        a = np.sort(own[np.isfinite(own)])[-4:]
        b = np.sort(enemy[np.isfinite(enemy)])[-4:]
        return (a[-1] if len(a) else 0) + .2*a.sum() - (b[-1] if len(b) else 0) - .2*b.sum()

    def search(self, board, player, depth, alpha, beta):
        _, wins = self.analyse(board,player)
        if len(wins):
            return 1.e8 + depth*1000
        _, blocks = self.analyse(board,3-player)
        if len(blocks) >= 2:
            return -1.e8 - depth*1000
        # 깊이가 끝나도 강제 방어가 남으면 최대 세 수 더 확인한다.
        if (depth <= 0 and not len(blocks)) or depth <= -3:
            return self.value(board,player)
        moves,_ = self.candidates(board,player,self.replies)
        if not len(moves):
            return 0.
        best = -np.inf
        for action in moves:
            y,x = divmod(int(action),20)
            board[y,x] = player
            value = -self.search(board,3-player,depth-1,-beta,-alpha)
            board[y,x] = 0
            best = max(best,value)
            alpha = max(alpha,value)
            if alpha >= beta:
                break
        return best

    def choose(self, board, player, qvalues):
        candidates, scores = self.candidates(board, player, self.width)
        if not len(candidates):
            raise ValueError('합법적인 착수 후보가 없습니다')
        _, immediate = self.analyse(board, player)
        if len(immediate):
            return int(max(immediate, key=lambda a: qvalues[a]))
        best_action, best_value = int(candidates[0]), -np.inf
        qscale = max(float(np.std(qvalues[np.isfinite(qvalues)])), 1.e-6)
        for action in candidates:
            y,x = divmod(int(action),20)
            board[y,x] = player
            # 이미 확보한 최선보다 좋아질 수 없는 가지는 루트에서도 중단한다.
            # 후보별 Q 보너스를 경계에 반영하여 최종 점수 비교와 일치시킨다.
            bonus = self.q_weight * float(qvalues[action])/qscale
            value = -self.search(board,3-player,self.depth-1,-np.inf,-best_value+bonus)
            board[y,x] = 0
            # Q값은 비슷한 전술 후보의 순위를 보조하며 강제 승패를 뒤집지 않는다.
            value += bonus
            if value > best_value:
                best_action,best_value = int(action),value
        return best_action
