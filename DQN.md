# 기보 기반 DQN 첫 모델

기존 지도학습 CNN 및 게임 실행 환경과 별도로 구성한 오프라인 DQN입니다.
기보에 실제로 기록된 행동만 학습하며, 자가 대국이나 온라인 탐험은 하지 않습니다.
새 모델이 기존 CNN보다 강하다고 가정하지 않고 대결 결과를 별도로 기록합니다.

## 파일

- `dqn_data.py`: Gomocup 2019 Freestyle ZIP → 경험 데이터. 원본 종료 상태를 재판정하며 잘못된 착수, 미완료 기록, 동일 기보를 제외합니다.
- `train_dqn.py`: MPS GPU에서 온라인/타깃 Q 네트워크 학습. GPU가 없으면 오류로 종료합니다.
- `evaluate_dqn.py`: 저장한 모델을 다시 불러와 기존 CNN 또는 무작위 상대와 대결합니다.
- `test_dqn.py`: 같은 플레이어 관점, 승패 보상, 불법 행동 마스킹, 종료 상태 목표값 검사.
- `models/dqn-v1/gomoku_dqn.pt`: 학습된 가중치와 실험 설정.
- `models/dqn-v1/training_report.json`: 데이터 분리 내역, 학습 이력, 검증 지표.
- `models/dqn-v1/evaluation_*.json`: 승패 및 대국별 실제 착수 기록.

## 구조와 학습

입력은 내 돌 +1 / 상대 돌 -1 / 빈칸 0의 `N×1×20×20` 텐서입니다.
3×3 Conv 32 → 64 → 64 (각 ReLU) → 1×1 Conv 1 → 400 Q값.
총 55,809 파라미터이며 마지막에 sigmoid/softmax를 사용하지 않습니다.

한 경험은 내 차례 s → 기보의 내 행동 a → 기보의 상대 행동 → 다음 내 차례 s′입니다.
내 착수로 이기면 +1, 상대 착수로 지면 -1, 진행 중 및 확인된 무승부는 0입니다.
종료 상태는 목표값 r만 사용하고, 나머지는 `r + 0.99 max Q_target(s′, 합법적 행동)`입니다.
타깃 네트워크는 500회 갱신마다 복사합니다. Adam 0.0001, Huber loss, 배치 128,
균등 경험 재현, gradient norm clipping 10, seed 2026, 5,000회 갱신입니다.
지도학습 사전학습이나 기존 CNN 가중치 이식은 사용하지 않습니다.

학습/검증은 대국 단위 해시 분리이며 완전히 동일한 기보는 제거합니다.
다른 대국의 공통 초반 배치까지 격리하는 분할은 아닙니다.
원본 데이터는 https://gomocup.org/results/gomocup-result-2019/ 의 Freestyle 기보를 사용합니다.

## 재현

```sh
uv venv .venv-dqn --python 3.12
uv pip install --python .venv-dqn/bin/python -r requirements-dqn.txt
.venv-dqn/bin/python -m unittest test_dqn -v
.venv-dqn/bin/python dqn_data.py --archive dataset/gomocup2019results.zip
.venv-dqn/bin/python train_dqn.py
.venv-dqn/bin/python evaluate_dqn.py --opponent cnn --games 10
.venv-dqn/bin/python evaluate_dqn.py --opponent random --games 10 --output models/dqn-v1/evaluation_random.json
```

CNN 대결 상대는 기존 `.venv`의 TensorFlow에서 별도 프로세스로 실행됩니다.
평가는 흑/백을 교대하며 각 쌍은 동일한 중앙 영역 두 수로 시작합니다.
DQN 평가는 CPU 추론을 사용하고 학습은 MPS GPU를 사용합니다.

## 해석

검증 기보 착수 일치율은 승률이 아닙니다. Bellman 오차도 대국 실력 자체가 아닙니다.
기록에 없는 행동의 Q값 과대평가는 오프라인 DQN의 한계이며,
이 모델은 학습 파이프라인을 검증한 첫 기준 모델입니다.
실제 대국 승률은 상대, 시작 배치, 판 수에 한정해 해석해야 합니다.

## 이번 실행 결과

- 하드웨어: Apple M1 Max, 24코어 GPU / 32GB 통합 메모리.
- MPS 순전파·역전파 검증 후 5,000회 GPU 학습 완료, 학습 루프 약 201초.
- 학습 8,710판 / 400,014개 경험, 검증 914판 / 42,376개 경험.
- 원본 10,224판 중 종료·유효성 검사 제외 114판, 완전히 같은 기보 중복 486판 제외.
- 검증 기보 다음 수 일치율: 4.75%. 기존 CNN의 노트북 검증 결과와는 분할·표본이 달라 직접 비교하지 않습니다.
- 기존 CNN 상대: 10판, 1승 9패 0무.
- 무작위 상대: 10판, 10승 0패 0무.
- 이 대결 조건에서는 기존 CNN보다 약합니다. 각 10판의 예비 평가이며 일반적인 승률을 보장하지 않습니다.
- 저장한 가중치를 다시 불러와 CPU 대결 및 MPS 추론을 확인했습니다.
- 기존 GUI는 여전히 원래 CNN을 사용합니다. DQN으로 자동 교체하지 않았습니다.
