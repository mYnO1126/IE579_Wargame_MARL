# MARL 설계 문서 (초안)

IE571 워게임 시뮬레이션의 각 troop을 에이전트로 보는 다중에이전트 강화학습(MARL) 설계.
이 문서는 구현 전 합의용 스펙이며, 차원·계수 등 숫자는 모두 **조정 가능한 초안**이다.

작성: 2026-06-17

---

## 0. 핵심 원칙

- **같은 동역학 유지**: 기존 시뮬의 물리(사격 Ph/Pk, LOS, 지형 이동비용, 관측)는 그대로
  "환경"으로 재사용한다. *규칙 기반 결정 부분만* 정책으로 대체한다. → sim-to-sim
  transfer gap 없음.
- **규모만 축소**: 작은 맵 + 소수 유닛 + 짧은 episode로 학습하고, 풀 시나리오는 평가/배포에만 쓴다.
- **별도의 다른 시뮬을 만들지 않는다.** 작은 맵은 같은 데이터/규칙의 축소판이어야 한다.


## 1. 목표 / 범위

- 각 유닛이 **(a) 이동 방향, (b) 타깃 선택, (c) 교전 여부**를 스스로 결정하도록 학습.
- 환경이 계산하는 것(정책이 건드리지 않음): 사격 명중/살상, 이동 거리·지형비용·주야,
  LOS·관측 갱신, 사망 처리.
- 1차 목표: 한 팀(청군)만 학습 vs 기존 스크립트 상대. 이후 self-play로 확장.


## 2. 전체 구조

```
rl/
  DESIGN.md         # 이 문서
  spawn.py          # 맵 랜덤 크롭 + 유닛 랜덤 스폰(유효 셀)
  obs.py            # 관측 벡터 생성
  reward.py         # 보상 계산
  env.py            # PettingZoo ParallelEnv 래퍼 (step/reset)
  wrappers.py       # 패딩/마스킹/정규화/파라미터공유 그룹
  train.py          # MAPPO 학습 루프
  policies/         # 체크포인트
```

- 기존 `modules/`는 **라이브러리로 import**해서 동역학을 재사용한다(시뮬 본체는 거의 안 건드림).
- 표준 다중에이전트 API: **PettingZoo Parallel API** (동시 이산시간 스텝 구조와 1:1).


## 3. 환경 (Environment)

### 3.1 맵
- **실제 `dem/slope/aspect/road/lake/wood/stream` 마스크에서 작은 영역을 랜덤 크롭**(예: 120×120 px).
  - 지형 분포는 실제와 동일 + 매 episode 다양 + 작아서 빠름.
  - 대안: 단순 합성 맵(단, 도로/숲/고지 등 대표 지형은 반드시 포함).
- `Map`을 크롭된 배열로 생성하거나, 크롭 offset만 들고 인덱싱.

### 3.2 스폰
- 양 팀 유닛을 **유효 셀에만** 랜덤 배치: `is_passable`(호수/급경사 제외), 겹침 방지는
  `grid_sample_no_overlap` 재사용.
- 병종 구성·수는 커리큘럼에 따라(§7). goal point도 팀별로 랜덤 지정(유효 셀).

### 3.3 결정 주기 (decision interval)
- 정책은 **매 N 시뮬분마다** 호출(권장 N=5~10). 그 사이 N분은 환경이 직전 행동을 유지하며
  물리만 진행(이동 지속, 사거리 들면 사격 등). → horizon 1/N로 축소, 학습 안정·고속화.

### 3.4 episode / 종료
- 작은 맵·소수 유닛, 최대 예 200~400 시뮬분(결정 20~80회).
- 종료: 한쪽 전멸 / 시간초과 / (옵션) 전 유닛 goal 도달.

### 3.5 에이전트
- 살아있는 각 유닛 = 1 에이전트. 사망 시 PettingZoo agent 제거.
- **파라미터 공유**: 같은 병종 범주끼리 정책망 공유(개별 322 정책 비현실적).


## 4. 관측 공간 (Observation) — per agent, egocentric

전부 **자기 기준 상대좌표 / 정규화**. 가변 개수는 top-k + 패딩 + valid 마스크.
병종은 5범주로 압축: `{TANK, ARMOR(apc), AT(atgm/rpg/recoilless), ARTY(indirect), INF}`.

| 블록 | 필드 | 차원(초안) |
|---|---|---|
| self | 병종범주 one-hot(5), 기동손상, 화력손상, range_km(정규화), 사거리내 적 flag | 9 |
| goal | 상대 dx, dy(정규화), 거리(정규화), 도달 flag | 4 |
| enemies top-k (k=5) | 각: 상대 dx, dy, 거리, 병종범주(5), 사거리내 flag, valid | 5×10 = 50 |
| allies top-k (k=5) | 각: 상대 dx, dy, 거리, 병종범주(5), 손상 flag, valid | 5×10 = 50 |
| 지형 패치 | 자기중심 11×11, 채널=[이동비용(정규화), 경사, 숲] | 11×11×3 = 363 |

- 지형 패치는 **지형 전술 학습에 사실상 필수**(고지/엄폐/도로). 초기엔 작은 CNN 또는 flatten MLP.
- top-k는 거리순. 적은 **탐지된(observed) 적만** 노출(부분관측, 기존 `observed` 시스템 사용).
- 절대위치(x/W, y/H)는 선택적으로 self에 추가 가능(맵 경계 인지용).


## 5. 행동 공간 (Action) — MultiDiscrete `[move, target, engage]`

| 헤드 | 의미 | 크기 |
|---|---|---|
| move | 8방향 + 정지 (또는 매크로: 목표로/적으로/엄폐로/대기) | 9 |
| target | 탐지된 적 top-k 중 선택 + none | k+1 = 6 |
| engage | 사격 / 대기(은폐 유지) | 2 |

- **절대 좌표를 출력하지 않는다**(학습 난이도 급증). 이동량·지형비용·주야는 환경이 적용.
- **마스킹 규칙**
  - target: valid 적만 선택 가능, 없으면 none 강제.
  - engage: 타깃 없거나 사거리 밖이면 "사격" 무효(자동 대기).
  - move: (옵션) 통행 불가 방향 마스킹.
- **초기 단순화**: engage는 "사거리 들면 자동 사격"으로 고정하고, hold-fire(은폐 전술)는
  사격=위치노출(`add_observed_troop`) 트레이드오프를 학습시킬 단계에서 도입.


## 6. 보상 (Reward) — per agent, per decision step

결과 보상 ≫ 셰이핑(보상 해킹 방지). 계수는 튜닝 대상.

- 가한 피해: 적 파괴 **+1.0**, 적 손상(M/F-kill) **+0.3**
- 받은 피해: 자기 사망 **−1.0**, 자기 손상 **−0.3**
- goal 접근: 거리 감소량에 소량 양(+) (예 `+0.01 × Δ거리_px`), 도달 시 보너스
- 종료 시 팀 보상: 승 **+W** / 패 **−W**, 또는 `(아군잔존 − 적잔존)/초기수` 비율
- (옵션) 무의미 이동·정체 페널티 소량

- 신용할당(credit assignment)이 어려운 MARL 특성상 **CTDE(중앙 critic)** 로 완화(§7).


## 7. 알고리즘 / 학습

- **MAPPO**(다중에이전트 PPO, CTDE) 기본값. 중앙 critic은 global state(전체 유닛 요약) 입력,
  실행은 분산(각자 local obs). 대안: 협동 가치분해 QMIX, 베이스라인 IPPO.
- **파라미터 공유**: 병종 범주별 정책(또는 팀+병종).
- **상대(opponent)**: ① 한 팀 학습 vs **frozen 스크립트 상대**(기존 `filter_priority` +
  pathfinding)로 시작 → ② **self-play**(과거 정책 풀과 대전).
- **커리큘럼**: 1v1 → 3v3 → 10v10 …, 맵·유닛 수·episode 길이 점증.
- 도구: **PyTorch로 직접 MAPPO 구현**(사용자 선택, torch 2.11 + CUDA). 환경은 외부 의존성
  없이 PettingZoo Parallel API 호환 인터페이스로 작성(필요 시 나중에 pettingzoo 래핑).
- 성능 목표: env 1 step ≪ 1ms(벡터화/병렬 env). 학습 중 **A*/flow-field 미사용**(정책이 직접
  방향 결정) → 기존 pathfinding 병목이 사라짐. pathfinding은 스크립트 상대에만 사용.


## 8. 기존 코드 연결점 (Integration)

**재사용(환경 동역학, 그대로):**
- `modules/map.py`: 지형/LOS(`is_visible`)/이동비용(`cost_map`, `movement_factor`)
- `modules/troop.py`: `Troop.fire`(Ph/Pk 사격 해결), `get_distance`,
  `calculate_movement_distance`(지형·주야 반영 이동량), `update_observation`/`find_observed_enemies`(관측)
- `modules/unit_definitions.py`: 병종 스펙·사거리·확률 함수

**정책으로 대체(결정 부분):**
- 이동 목적지: `update_troop_location_improved`의 dest 결정 / `TacticalManager` → move 액션
- 타게팅: `assign_targets` / `assign_target` / `filter_priority` → target 액션
- 사격 개시: 자동 → engage 액션(또는 초기엔 자동 유지)

**학습 시 우회/대체:**
- `PLACEMENT`/`TIMELINE`(고정 배치·페이즈) → 랜덤 스폰·랜덤 goal로 대체(풀 시나리오 평가 때만 원본 사용)
- A*/flow-field → 학습 중 미사용


## 9. 마일스톤

1. **환경화**: 결정 함수를 액션 입력으로 치환 + PettingZoo 래핑. 랜덤 정책으로 1 episode 통과 확인.
2. **고속화**: 작은 맵·소수 유닛·결정주기 → step 시간 측정·벡터화.
3. **단일팀 학습 vs 스크립트**: MAPPO + 병종 공유로 청군 학습, 학습곡선 확인.
4. **커리큘럼 확대**: 유닛·맵·길이 점증.
5. **self-play**: 양팀 학습(정책 풀).
6. **풀 시뮬 배포 + 베이스라인 비교**(§10): 학습 정책을 풀 시나리오에 끼워 규칙기반과 비교.


## 10. 평가 / 베이스라인 비교 (★ 필수 목표)

이 프로젝트의 최종 산출물은 **"MARL 정책 vs 기존 규칙기반 행동"의 정량 비교**다. 설계 전반이
이 비교를 깨끗하게 할 수 있도록 맞춰져야 한다.

- **규칙기반 경로를 절대 훼손하지 않는다.** 기존 결정 로직(`assign_targets`/`filter_priority`,
  `update_troop_location_improved`/`TacticalManager`)은 그대로 보존(코드 삭제 금지 원칙).
  `rl/`은 `modules/`를 라이브러리로 재사용하는 **별도 경로**이지 기존 코드를 덮어쓰지 않는다.
- **결정만 교체하는 스위치(decision provider).** 풀 시뮬(`main.py`, PLACEMENT/TIMELINE)을
  ①규칙기반 또는 ②학습 정책으로 구동. 물리·사격(Ph/Pk)·LOS·지형·관측은 양쪽 100% 동일.
- **시드 일치.** 전투 난수·스폰 seed를 동일하게 → 결과 차이가 오직 "정책" 때문임을 보장.
- **동일 지표.** 기존 `History` 산출물(시간별 팀 병력 곡선, 사상자, 승패, 종료시각, plot.png/CSV)을
  그대로 사용해 두 모드를 나란히 비교.
- **다중 시드 통계.** 단판은 노이즈 큼 → N seed 평균±표준편차로 비교.
- **스케일 불변 관측 전제.** obs가 egocentric + top-k(고정 차원)라, 작은 크롭에서 학습한 정책을
  풀 시나리오(다수 유닛·큰 맵)에 그대로 배포해 비교 가능. (이 성질을 깨지 말 것.)


## 11. 미해결 / 결정 필요

- goal point는 누가? (고정/랜덤/상위 commander 정책) — 1차는 랜덤 고정.
- hold-fire(은폐 전술) 도입 시점.
- 보상 계수·정규화 스케일.
- 관측에 절대위치 포함 여부.
- 탄약/보급(현재 시뮬에서 비활성) 학습 범위 포함 여부.
- 중앙 critic의 global state 표현(전체 유닛 요약 방식).
