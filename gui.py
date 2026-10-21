from pathlib import Path
from ursina import *
import numpy as np
from tensorflow.keras.models import load_model
from gomoku import Board, Gomoku

# 실행 폴더와 관계없이 저장소의 모델을 읽는다. 추론에는 과거 optimizer 상태가 필요 없다.
model = load_model(Path(__file__).resolve().parent / 'models/20201213_202430.h5', compile=False)

# 원본에서 사용한 Ursina 3.2.2로 대결 화면을 구성한다.
app = Ursina()

window.title = 'Omok AI — You: Black / AI: White'
window.borderless = False
window.color = color._50

w, h = 20, 20
camera.orthographic = True
camera.fov = 23
# 카메라를 판 앞에 두어 현재 Ursina에서도 평면과 같은 깊이에 놓이지 않게 한다.
camera.position = (w//2, h//2, -20)

board = Board(w=w, h=h)
board_buttons = [[None for x in range(w)] for y in range(h)]
game = Gomoku(board=board)

Entity(model='quad', scale=(22, 22), position=(9.5, 9.5, 0.2), color=color.white)
Entity(model=Grid(w+1, h+1), scale=w+1, color=color.black, x=w//2-0.5, y=h//2-0.5, z=0.1)

for y in range(h):
    for x in range(w):
        b = Button(parent=scene, position=(x, y), color=color.clear, model='circle', scale=0.9) # start from bottom left
        board_buttons[y][x] = b

        def on_mouse_enter(b=b):
            if b.collision:
                b.color = color._100
        def on_mouse_exit(b=b):
            if b.collision:
                b.color = color.clear

        b.on_mouse_enter = on_mouse_enter
        b.on_mouse_exit = on_mouse_exit

        def on_click(b=b):
            if game.won_player or game.board.board[h - int(b.y) - 1, int(b.x)] != 0:
                return
            # player turn
            b.text = '1'
            b.color = color.black
            b.collision = False

            game.put(x=int(b.position.x), y=int(h - b.position.y - 1)) # start from top left

            won_player = game.check_won()

            if won_player > 0:
                end_session(won_player)
                return

            if not np.any(game.board.board == 0):
                end_session(0)
                return

            game.next()

            # cpu turn
            input = game.board.board.copy()
            # 학습과 같은 관점: 현재 AI의 돌은 +1, 상대의 돌은 -1이다.
            player = game.current_player
            input = np.where(input == player, 1, np.where(input == 0, 0, -1))
            input = np.expand_dims(input, axis=(0, -1)).astype(np.float32)

            output = model(input, training=False).numpy().squeeze()
            output = output.reshape((h, w))
            # 합법적인 빈칸만 선택하여 기존 돌을 덮어쓰지 않는다.
            output[game.board.board != 0] = -np.inf
            output_y, output_x = np.unravel_index(np.argmax(output), output.shape)

            game.put(x=output_x, y=output_y)

            board_buttons[h - output_y - 1][output_x].text = '2'
            board_buttons[h - output_y - 1][output_x].text_color = color.black
            board_buttons[h - output_y - 1][output_x].color = color.white
            board_buttons[h - output_y - 1][output_x].collision = False

            won_player = game.check_won()

            if won_player > 0:
                end_session(won_player)
                return

            if not np.any(game.board.board == 0):
                end_session(0)
                return

            game.next()

            print(game.board)

        b.on_click = on_click

def end_session(won_player):
    Panel(z=1, scale=10, model='quad')
    message = 'Draw!' if won_player == 0 else f'Player {won_player} won!'
    t = Text(message, scale=3, origin=(0, 0), background=True)
    t.create_background(padding=(.5,.25), radius=Text.size/2)

app.run()