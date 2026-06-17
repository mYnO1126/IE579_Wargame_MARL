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

#!CLAUDE 지형 관측: 11x11 패치(363차원, CNN 필요) 대신 8방향 압축 특징으로 교체 → MLP로 충분.
#         각 방향 = _MOVES 1..8 과 동일 순서. 방향마다 {이동가능거리, 평균 험준함, 고도변화} 3개.
_DIR8 = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
_LOOK = 12             # 각 방향으로 직선 탐색할 셀 수
TERR_FEATS = 8 * 3     # 24

# self(11) + goal(5) + enemies(K*10) + allies(K*10) + terrain(8dir*3=24)
OBS_DIM = 11 + 5 + ENEMY_K * 10 + ALLY_K * 10 + TERR_FEATS


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

    # --- self (11) ---
    self_block = np.zeros(11, dtype=np.float32)
    self_block[0:5] = _cat_onehot(troop.type)
    self_block[5] = 1.0 if getattr(troop, "status", None) and troop.status.value == "mobility_damaged" else 0.0
    self_block[6] = 1.0 if getattr(troop, "status", None) and troop.status.value == "firepower_damaged" else 0.0
    self_block[7] = min(troop.range_km / 4.0, 1.0)
    enemy_order = _topk_by_distance(troop, enemies_pool, ENEMY_K)
    self_block[8] = 1.0 if (enemy_order and troop.get_distance(enemy_order[0]) <= troop.range_km) else 0.0
    # 절대위치(맵 경계 인지용) — 정규화
    self_block[9] = troop.coord.x / W
    self_block[10] = troop.coord.y / H
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

    # --- terrain: 8방향 직선 특징 (이동가능거리 / 평균 험준함 / 고도변화) ---
    # 각 방향으로 _LOOK 셀까지 직진하며 통행 가능한 만큼의 거리·평균 이동비용·끝점 고도차를 요약.
    cx, cy = int(round(troop.coord.x)), int(round(troop.coord.y))
    z0 = float(cm.dem_arr[cy, cx]) if (0 <= cx < W and 0 <= cy < H) else 0.0
    terr = np.zeros(TERR_FEATS, dtype=np.float32)
    for di, (ddx, ddy) in enumerate(_DIR8):
        reach = 0
        cost_sum = 0.0
        end_z = z0
        for s in range(1, _LOOK + 1):
            x, y = cx + ddx * s, cy + ddy * s
            if not (0 <= x < W and 0 <= y < H):
                break
            c = cm.cost_map[y, x]
            if not np.isfinite(c):     # 통행 불가(호수/급경사) → 거기서 막힘
                break
            reach = s
            cost_sum += min(float(c), 20.0)
            end_z = float(cm.dem_arr[y, x])
        off = di * 3
        terr[off + 0] = reach / _LOOK                                    # 이동 가능 거리
        terr[off + 1] = (cost_sum / reach / 20.0) if reach > 0 else 1.0  # 평균 험준함(0~1)
        terr[off + 2] = float(np.clip((end_z - z0) / 50.0, -1.0, 1.0))   # 고도 변화(+=상승)
    parts.append(terr)

    return np.concatenate(parts).astype(np.float32), enemy_order
