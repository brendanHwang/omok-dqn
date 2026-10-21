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

## DQN + 전술 탐색 개선

`tactical_dqn.py`는 가중치를 바꾸지 않고 행동 선택을 개선합니다.
즉시 승리/방어를 우선하고, 4 ply(내 수·상대 수·내 수·상대 수)의 negamax와
alpha-beta 가지치기를 적용합니다. 강제 방어는 최대 3 ply 추가 확인합니다.
루트 후보 12개, 각 응수 8개이며, 전술 점수에 작은 DQN Q 보너스(0.03)를 더합니다.
깊이와 탐색 폭이 제한되므로 모든 가능한 수를 검사하는 완전 탐색은 아닙니다.

개발 배치(seed 2026, 10개 시작 배치를 흑/백 교대)에서 3 ply와 4 ply 모두 10승 10패였습니다.
설정을 고정한 뒤 다른 시작 배치(seed 9000, 10개 시작 배치를 흑/백 교대)에서 검증했습니다.
모든 상대는 기존 CNN이며, 동일한 시작 배치와 같은 색을 사용했습니다.

| 버전 | 새 배치 20판 |
|---|---:|
| 순수 DQN | 5승 15패 |
| 탐색만 (Q 보너스 0) | 13승 7패 |
| DQN + 탐색 | 16승 4패 |

이는 재학습 효과가 아니라 탐색을 결합한 에이전트의 개선입니다.
이번 표본에서는 Q 보너스를 쓴 버전이 더 이겼지만, 작은 표본으로 일반적인 기여를 확정할 수 없습니다.
탐색 없는 CNN과 탐색을 사용하는 에이전트 비교이며, 같은 계산량의 모델 우열 비교는 아닙니다.
게임 60판의 착수·승패를 기존 Gomoku 규칙 코드로 재검증했고 가중치 체크섬도 유지됐습니다.

검증 당시 DQN+탐색의 평균 선택 시간은 약 252ms였습니다.
이후 루트 가지치기 최적화는 별도 20개 국면에서 선택한 수를 유지하며 합산 7.46초 → 3.07초로 줄였습니다.
저장된 대국 평가 시간은 이 최적화 전 수치이며, 단말/부하에 따라 달라집니다.
관련 전술·데이터 테스트 7개를 통과했습니다.

- `models/tactical-dqn-v1/agent.json`: 고정한 구성과 원본 DQN 체크섬.
- `models/tactical-dqn-v1/comparison.json`: 검증 비교 요약.
- `models/tactical-dqn-v1/evaluation_unseen.json`: DQN+탐색의 대국별 착수.
- `models/tactical-dqn-v1/evaluation_search_only.json`: 탐색 단독 대조군.
- `models/tactical-dqn-v1/baseline_unseen.json`: 같은 조건의 순수 DQN.

```sh
.venv-dqn/bin/python evaluate_dqn.py --agent tactical-dqn --opening-seed 9000 --games 20 --output models/tactical-dqn-v1/recheck.json
```

기존 게임 GUI는 여전히 CNN을 사용하며 자동 교체하지 않았습니다.
