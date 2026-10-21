# omok-dqn

20×20 오목에서 **기보 기반 DQN과 전술 탐색**을 실험하는 프로젝트입니다.
학습된 모델, 대결 평가 코드, 후보 점수를 볼 수 있는 게임 화면을 제공합니다.

[kairess/omok-ai](https://github.com/kairess/omok-ai)를 기반으로 게임 UI 개선,
Apple GPU DQN 학습, 전술 탐색 및 비교 평가를 추가했습니다. 원본 Git 이력을 유지합니다.

## 포함된 기능

- 사람 vs 기존 CNN 오목 게임: 재시작, 차례 표시, 마지막 AI 착수 표시.
- `AI preview`: 내 차례와 AI 차례의 상위 3개 후보 및 모델 점수 표시.
- 같은 Gomocup 기보로 학습한 오프라인 DQN 체크포인트.
- 즉시 승리·방어, 4수 탐색 및 강제 방어 연장을 결합한 DQN 에이전트.
- 순수 DQN / 탐색 단독 / DQN+탐색의 CNN 상대 대결 평가와 실제 대국 기록.

**게임 GUI는 기존 CNN을 사용합니다. DQN+탐색 버전은 평가 명령으로 실행합니다.**
모델 점수는 승률이 아니며, 플레이 중 추가 학습이나 대국 정보 외부 전송 기능은 없습니다.

## 다운로드

```sh
git clone https://github.com/brendanHwang/omok-dqn.git
cd omok-dqn
```

Git 없이 받으려면 GitHub의 **Code → Download ZIP**을 사용하세요.
학습된 CNN과 DQN은 `models/`에 포함되어 있어 대결 실행을 위해 재학습할 필요가 없습니다.
원본 기보, 전처리 데이터, 가상환경은 포함하지 않습니다.

## 게임 실행

검증 환경: **macOS Apple Silicon, Python 3.12**. 다른 OS의 GUI 동작은 검증하지 않았습니다.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-play.txt
.venv/bin/python gui.py
```

- 검은 돌이 사용자, 흰 돌이 AI입니다.
- `AI preview: On`을 켜면 현재 플레이어 관점의 후보 3개와 점수가 보입니다.
- 사용자 차례에는 원하는 빈 교점에 직접 착수합니다.
- AI 후보를 확인한 뒤 `Play AI move`를 누르면 AI가 착수합니다.
- `New game`으로 다시 시작합니다.

## DQN 환경과 평가

게임 실행 환경과 분리합니다. 아래 평가 명령은 CPU 추론을 사용하므로 MPS가 필요하지 않습니다.
CNN 상대 프로세스를 위해 위의 `.venv`도 구성해야 합니다.

```sh
python3.12 -m venv .venv-dqn
.venv-dqn/bin/python -m pip install -r requirements-dqn.txt
.venv-dqn/bin/python -m unittest test_dqn test_tactical_dqn -v

# DQN + 전술 탐색 vs 기존 CNN
.venv-dqn/bin/python evaluate_dqn.py --agent tactical-dqn --opening-seed 9000 --games 20 --output results/tactical.json

# 같은 배치의 순수 DQN / 탐색 단독 대조군
.venv-dqn/bin/python evaluate_dqn.py --agent dqn --opening-seed 9000 --games 20 --output results/dqn.json
.venv-dqn/bin/python evaluate_dqn.py --agent tactical-only --opening-seed 9000 --games 20 --output results/search-only.json
```

### 기록된 예비 평가

개발에 사용하지 않은 10개 시작 배치에서 흑·백을 바꾼 20판입니다. 상대는 원본 CNN입니다.

| 방식 | 승 | 패 |
|---|---:|---:|
| 순수 DQN | 5 | 15 |
| 탐색 단독 | 13 | 7 |
| DQN + 전술 탐색 | 16 | 4 |

DQN 가중치는 동일하며 **탐색을 결합한 에이전트의 개선**입니다.
탐색 없는 CNN과 계산량이 같지 않고, 작은 표본이므로 일반적인 승률을 보장하지 않습니다.
상세 대국 기록은 `models/tactical-dqn-v1/`에 있습니다.

## DQN 재학습 (Mac GPU)

Apple Silicon의 PyTorch MPS를 사용합니다. GPU가 없으면 학습을 자동으로 CPU로 대체하지 않고 종료합니다.

1. [Gomocup 2019](https://gomocup.org/results/gomocup-result-2019/)에서 결과 ZIP을 다운로드합니다.
2. 아래 명령으로 Freestyle 기보를 전처리하고 학습합니다.

```sh
.venv-dqn/bin/python dqn_data.py --archive /path/to/gomocup2019results.zip
.venv-dqn/bin/python train_dqn.py --output models/my-dqn
```

보상, 네트워크 구조, 학습 수량, 한계, 재현 절차는 [DQN.md](DQN.md)에 정리했습니다.
모델 출력은 위치별 Q값이며 기보 모방 CNN의 sigmoid 점수와 다릅니다.

## 파일 안내

| 파일 | 역할 |
|---|---|
| `gui.py` | CNN 게임 화면 및 양쪽 플레이어 후보 미리보기 |
| `gomoku.py` | 착수·차례·승패 규칙 |
| `dqn_data.py` | 기보 → 상태·행동·보상·다음 상태 |
| `train_dqn.py` | 온라인/타깃 Q 네트워크의 GPU 학습 |
| `tactical_dqn.py` | DQN 점수와 전술 탐색 결합 |
| `evaluate_dqn.py` | 고정 모델 간 대결 및 기보 저장 |
| `test_dqn.py`, `test_tactical_dqn.py` | 데이터·학습 목표·전술 검사 |
| `create_dataset.py`, `read_dataset.py` | 원본 지도학습 CNN 데이터 준비·확인 |
| `train_cnn.ipynb` | 원본 CNN 학습 코드와 기존 실행 로그 |
| `train_neat.py`, `config` | 원작자의 별도 NEAT 실험. 현재 실행 경로에서 사용하지 않음 |
| `models/` | 원본 CNN, 학습된 DQN, 설정·평가 기록 |

## 출처

- 원본 코드와 CNN 모델: [kairess/omok-ai](https://github.com/kairess/omok-ai)
- 학습 기보: [Gomocup 결과 자료](https://gomocup.org/results/)
- 원본 저장소에는 별도 LICENSE 파일이 없습니다. 이 저장소에서도 원본 코드·모델에 임의의 새 라이선스를 부여하지 않습니다.
