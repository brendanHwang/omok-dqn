from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from ursina import *
import numpy as np
from tensorflow.keras.models import load_model
from gomoku import Board, Gomoku

# 실행 폴더와 관계없이 저장소의 모델을 읽는다. 추론에는 과거 optimizer 상태가 필요 없다.
model = load_model(Path(__file__).resolve().parent / 'models/20201213_202430.h5', compile=False)

# 원본에서 사용한 Ursina 3.2.2로 대결 화면을 구성한다.
app = Ursina()
window.title = 'Omok'
window.borderless = False
window.color = color.rgb(33, 43, 48)
window.fps_counter.enabled = False
window.exit_button.enabled = False
window.cog_button.enabled = False

camera.orthographic = True
camera.fov = 26
# 카메라를 판 앞에 두어 현재 Ursina에서도 평면과 같은 깊이에 놓이지 않게 한다.
camera.position = (13.8, 9.5, -30)


class OmokGame(Entity):
    """대국 화면과 비동기 추론의 완료 처리를 담당한다. 규칙은 Gomoku가 관리한다."""

    def __init__(self):
        super().__init__()
        self.board = Board(w=20, h=20)
        self.game = Gomoku(board=self.board)
        self.buttons = []
        self.stone_details = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.ai_future = None
        self.finished = False
        self.ink = color.rgb(232, 237, 235)
        self.muted = color.rgb(154, 173, 176)
        self.accent = color.rgb(113, 206, 189)
        self._build_board()
        self._build_panel()

    def _build_board(self):
        Entity(model='quad', position=(9.65, 9.3, .6), scale=22.1,
               color=color.rgba(0, 0, 0, 65))
        Entity(model='quad', position=(9.5, 9.5, .4), scale=21.8,
               color=color.rgb(110, 78, 49))
        Entity(model='quad', position=(9.5, 9.5, .3), scale=21.35,
               color=color.rgb(211, 173, 115))
        # 격자와 돌 모두 같은 교점을 사용한다. 화면 모양은 모델 입력에 영향을 주지 않는다.
        vertices = []
        for i in range(20):
            vertices.extend([(i, 0, 0), (i, 19, 0), (0, i, 0), (19, i, 0)])
        Entity(model=Mesh(vertices=vertices, triangles=[(i, i + 1) for i in range(0, len(vertices), 2)],
                          mode='line', thickness=1),
               color=color.rgb(114, 84, 50), z=.15)
        for i in range(20):
            Text(chr(65 + i), parent=scene, position=(i, 19.8, -.05),
                 origin=(0, 0), scale=19, color=color.rgb(105, 75, 43))
            Text(str(i + 1), parent=scene, position=(-.8, 19 - i, -.05),
                 origin=(0, 0), scale=19, color=color.rgb(105, 75, 43))
        for x in (3, 9, 16):
            for y in (3, 9, 16):
                Entity(model='circle', position=(x, y, .1), scale=.13,
                       color=color.rgb(100, 72, 43))
        # AI의 마지막 착수에만 테두리를 겹쳐 표시한다.
        self.last_ai_outline = Entity(model=Circle(64, mode='line', thickness=3),
                                      color=self.accent, scale=1.02, z=-.12, enabled=False)
        for y in range(20):
            row = []
            for x in range(20):
                b = Button(parent=scene, position=(x, y), model='circle',
                           scale=.88, color=color.clear)  # start from bottom left
                b.on_mouse_enter = lambda b=b: self._hover(b, True)
                b.on_mouse_exit = lambda b=b: self._hover(b, False)
                b.on_click = lambda x=x, y=y: self.play_human(x, 19 - y)
                row.append(b)
            self.buttons.append(row)

    def _build_panel(self):
        Text('Omok', position=(-.57, .47), scale=1.65, color=self.ink)
        Text('20 x 20 / Freestyle', position=(-.57, -.445), scale=.85, color=self.muted)
        self.panel = Entity(parent=camera.ui, x=.47)
        Text('Your match', parent=self.panel, position=(-.13, .39), scale=1.15, color=self.muted)
        Entity(parent=self.panel, model='quad', position=(.015, .34),
               scale=(.29, .001), color=color.rgb(72, 91, 96))
        self.turn_label = Text('Your turn', parent=self.panel, position=(-.13, .29),
                               scale=1.6, color=self.accent)
        self.turn_detail = Text('Place a black stone.', parent=self.panel,
                                position=(-.13, .24), scale=.85, color=self.ink)
        for y, stone_color, label in ((.13, color.rgb(27, 30, 33), 'You / Black'),
                                      (.065, color.rgb(244, 244, 235), 'AI / White')):
            Entity(parent=self.panel, model='circle', position=(-.112, y, -.01),
                   scale=.027, color=stone_color)
            Text(label, parent=self.panel, position=(-.08, y + .01), scale=.9, color=self.ink)
        self.move_label = Text('Moves   0', parent=self.panel, position=(-.13, -.045),
                               scale=1, color=self.ink)
        self.last_move_label = Text('Last AI move   --', parent=self.panel,
                                    position=(-.13, -.10), scale=.85, color=self.muted)
        self.response_label = Text('', parent=self.panel, position=(-.13, -.145),
                                   scale=.8, color=self.muted)
        Text('Five in a row wins.\nGreen ring: last AI move.', parent=self.panel,
             position=(-.13, -.25), scale=.78, color=self.muted)
        self.restart_button = Button(parent=self.panel, text='New game',
                                     position=(.015, -.37), scale=(.29, .065),
                                     color=self.accent, text_color=color.rgb(27, 45, 46),
                                     highlight_color=color.rgb(147, 224, 210),
                                     on_click=self.restart_game)

    def _hover(self, button, entered):
        if button.collision:
            button.color = (color.rgba(25, 31, 32, 65)
                            if entered and self.game.current_player == 1 and not self.finished
                            else color.clear)

    def _show_stone(self, x, y, player):
        button = self.buttons[19 - y][x]
        button.color = color.rgb(30, 33, 36) if player == 1 else color.rgb(248, 247, 237)
        button.collision = False
        # 작은 하이라이트로 돌을 구분하며 숫자 표시는 사용하지 않는다.
        detail = Entity(parent=button, model='circle', position=(-.13, .15, -.05),
                        scale=.48, color=(color.rgb(43, 47, 50) if player == 1
                                          else color.rgb(255, 255, 250)))
        self.stone_details.append(detail)
        self.move_label.text = f'Moves   {np.count_nonzero(self.board.board)}'

    def restart_game(self):
        # 대기 중인 AI 처리를 취소하여 새 대국에 이전 착수가 섞이지 않게 한다.
        # 이미 계산 중이면 결과를 버린다. 작업 스레드는 화면이나 대국 상태를 수정하지 않는다.
        if self.ai_future is not None:
            self.ai_future.cancel()
            self.ai_future = None
        self.game.reset()
        self.finished = False
        for detail in self.stone_details:
            destroy(detail)
        self.stone_details.clear()
        for row in self.buttons:
            for button in row:
                button.color = color.clear
                button.collision = True
        self.last_ai_outline.enabled = False
        self.turn_label.text = 'Your turn'
        self.turn_detail.text = 'Place a black stone.'
        self.move_label.text = 'Moves   0'
        self.last_move_label.text = 'Last AI move   --'
        self.response_label.text = ''

    def play_human(self, x, y):
        if self.finished or self.game.current_player != 1 or self.board.board[y, x] != 0:
            return
        # player turn
        self.game.put(x=x, y=y)  # start from top left
        self._show_stone(x, y, 1)
        if self._check_end():
            return
        self.game.next()
        self.turn_label.text = 'AI thinking'
        self.turn_detail.text = 'White is choosing a move.'
        # 차례 표시가 먼저 그려진 뒤 AI 추론을 실행한다.
        # 추론에는 복사본만 전달하여 화면 갱신과 재시작 버튼을 막지 않는다.
        self.ai_future = self.executor.submit(self._predict_move, self.board.board.copy())

    @staticmethod
    def _predict_move(board):
        started = perf_counter()
        # cpu turn
        # 학습과 같은 관점: 현재 AI의 돌은 +1, 상대의 돌은 -1이다.
        input = np.where(board == 2, 1, np.where(board == 0, 0, -1))
        input = np.expand_dims(input, axis=(0, -1)).astype(np.float32)
        output = model(input, training=False).numpy().reshape((20, 20))
        # 합법적인 빈칸만 선택하여 기존 돌을 덮어쓰지 않는다.
        output[board != 0] = -np.inf
        y, x = np.unravel_index(np.argmax(output), output.shape)
        return int(x), int(y), perf_counter() - started

    def update(self):
        if self.ai_future is None or not self.ai_future.done():
            return
        future = self.ai_future
        self.ai_future = None
        try:
            x, y, elapsed = future.result()
        except Exception as error:
            self.finished = True
            self.turn_label.text = 'AI unavailable'
            self.turn_detail.text = 'Select New game to retry.'
            print(f'AI inference failed: {error}')
            return
        self.game.put(x=x, y=y)
        self._show_stone(x, y, 2)
        self.last_ai_outline.position = (x, 19 - y, -.12)
        self.last_ai_outline.enabled = True
        self.last_move_label.text = f'Last AI move   {chr(65 + x)}{y + 1}'
        self.response_label.text = f'AI response   {elapsed:.2f} s'
        if self._check_end():
            return
        self.game.next()
        self.turn_label.text = 'Your turn'
        self.turn_detail.text = 'Place a black stone.'

    def _check_end(self):
        winner = self.game.check_won()
        if winner == 0 and np.any(self.board.board == 0):
            return False
        self.finished = True
        # 종료 안내는 차례 표시 영역에 남겨 재시작 버튼을 가리지 않는다.
        if winner == 1:
            self.turn_label.text = 'You won!'
        elif winner == 2:
            self.turn_label.text = 'AI won'
        else:
            self.turn_label.text = 'Draw'
        self.turn_detail.text = 'Select New game to play again.'
        return True

    def on_destroy(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


game_view = OmokGame()
app.run()
