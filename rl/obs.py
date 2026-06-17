# obs.py
# 에이전트별 egocentric 관측 벡터 생성 (DESIGN.md §4).
# 반환: (vec[float32], enemy_order[list[Troop]])
#   enemy_order = 관측에 인코딩된 적군 순서. env 가 target 행동 인덱스를 이 순서에 매핑한다.

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import math
import numpy as np

from modules.unit_definitions import UnitType

ENEMY_K = 5
ALLY_K = 5
PATCH = 11           # 자기 중심 지형 패치 한 변(픽셀)
_HALF = PATCH // 2

# self(9) + goal(5) + enemies(K*10) + allies(K*10) + terrain(PATCH*PATCH*3)
OBS_DIM = 9 + 5 + ENEMY_K * 10 + ALLY_K * 10 + PATCH * PATCH * 3


def unit_cat(t):
    """병종 5범주: 0 전차 / 1 장갑(APC) / 2 대전차 / 3 간접화력 / 4 보병·기타."""
    if t == UnitType.TANK:
        return 0
    if t == UnitType.APC:
        return 1
    if UnitType.is_anti_tank(t):
        return 2
    if UnitType.is_indirect_fire(t):
        return 3
    return 4


def _cat_onehot(t):
    v = np.zeros(5, dtype=np.float32)
    v[unit_cat(t)] = 1.0
    return v


def _topk_by_distance(troop, others, k):
    """troop 기준 거리순 정렬 후 상위 k. (살아있는 것만)"""
    alive = [o for o in others if o.alive and o is not troop]
    alive.sort(key=lambda o: troop.get_distance(o))
    return alive[:k]


def build_observation(troop, troop_list, cm):
    """troop 한 기의 관측 벡터와 enemy_order 를 만든다.
    goal 은 기존 시나리오와 동일하게 troop.fixed_dest 를 사용한다(RED만 보유, BLUE는 None)."""
    W, H = cm.width, cm.height
    dmax = math.hypot(W, H)

    if troop.team == "blue":
        enemies_pool = troop_list.red_observed
        allies_pool = troop_list.blue_troops
    else:
        enemies_pool = troop_list.blue_observed
        allies_pool = troop_list.red_troops

    parts = []

    # --- self (9) ---
    self_block = np.zeros(9, dtype=np.float32)
    self_block[0:5] = _cat_onehot(troop.type)
    self_block[5] = 1.0 if getattr(troop, "status", None) and troop.status.value == "mobility_damaged" else 0.0
    self_block[6] = 1.0 if getattr(troop, "status", None) and troop.status.value == "firepower_damaged" else 0.0
    self_block[7] = min(troop.range_km / 4.0, 1.0)
    enemy_order = _topk_by_distance(troop, enemies_pool, ENEMY_K)
    self_block[8] = 1.0 if (enemy_order and troop.get_distance(enemy_order[0]) <= troop.range_km) else 0.0
    parts.append(self_block)

    # --- goal (5): has_goal + 상대 벡터/거리(정규화) + 도달 flag ---
    # RED(공격)만 fixed_dest 보유, BLUE(방어)는 None → has_goal=0, 나머지 0.
    goal = getattr(troop, "fixed_dest", None)
    if goal is not None:
        gdx = goal.x - troop.coord.x
        gdy = goal.y - troop.coord.y
        gdist = math.hypot(gdx, gdy)
        goal_block = np.array([
            1.0, gdx / dmax, gdy / dmax, min(gdist / dmax, 1.0),
            1.0 if gdist < 5.0 else 0.0,
        ], dtype=np.float32)
    else:
        goal_block = np.zeros(5, dtype=np.float32)
    parts.append(goal_block)

    # --- enemies top-k (K*10) ---
    enemy_block = np.zeros(ENEMY_K * 10, dtype=np.float32)
    for i, e in enumerate(enemy_order):
        d = troop.get_distance(e)
        off = i * 10
        enemy_block[off + 0] = (e.coord.x - troop.coord.x) / dmax
        enemy_block[off + 1] = (e.coord.y - troop.coord.y) / dmax
        enemy_block[off + 2] = min((d * 100.0) / dmax, 1.0)  # d(km)->px
        enemy_block[off + 3:off + 8] = _cat_onehot(e.type)
        enemy_block[off + 8] = 1.0 if d <= troop.range_km else 0.0
        enemy_block[off + 9] = 1.0  # valid
    parts.append(enemy_block)

    # --- allies top-k (K*10) ---
    ally_order = _topk_by_distance(troop, allies_pool, ALLY_K)
    ally_block = np.zeros(ALLY_K * 10, dtype=np.float32)
    for i, a in enumerate(ally_order):
        d = troop.get_distance(a)
        off = i * 10
        ally_block[off + 0] = (a.coord.x - troop.coord.x) / dmax
        ally_block[off + 1] = (a.coord.y - troop.coord.y) / dmax
        ally_block[off + 2] = min((d * 100.0) / dmax, 1.0)
        ally_block[off + 3:off + 8] = _cat_onehot(a.type)
        ally_block[off + 8] = 1.0 if getattr(a, "status", None) and "damaged" in a.status.value else 0.0
        ally_block[off + 9] = 1.0
    parts.append(ally_block)

    # --- terrain patch (PATCH*PATCH*3): [이동비용, 경사, 숲] ---
    cx, cy = int(round(troop.coord.x)), int(round(troop.coord.y))
    cost = np.full((PATCH, PATCH), 1.0, dtype=np.float32)   # 범위 밖=최대비용
    slope = np.zeros((PATCH, PATCH), dtype=np.float32)
    wood = np.zeros((PATCH, PATCH), dtype=np.float32)
    for j in range(PATCH):
        for i in range(PATCH):
            x = cx + (i - _HALF)
            y = cy + (j - _HALF)
            if 0 <= x < W and 0 <= y < H:
                c = cm.cost_map[y, x]
                cost[j, i] = 1.0 if not np.isfinite(c) else min(float(c), 20.0) / 20.0
                slope[j, i] = min(float(cm.slope_arr[y, x]) / 90.0, 1.0)
                wood[j, i] = 1.0 if cm.wood_mask[y, x] else 0.0
    parts.append(np.stack([cost, slope, wood], axis=-1).reshape(-1))

    return np.concatenate(parts).astype(np.float32), enemy_order
