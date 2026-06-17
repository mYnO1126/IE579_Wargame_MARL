# spawn.py
# 학습용 맵 생성(실제 맵의 작은 랜덤 크롭) + 유닛 랜덤 스폰.
# 기존 시뮬의 Map / Troop 동역학을 그대로 재사용한다 (DESIGN.md §3).

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from modules.map import Map, Coord
from modules.troop import Troop

_FULL_MAP = None


def load_full_map():
    """전체 맵을 한 번만 로드해서 캐시."""
    global _FULL_MAP
    if _FULL_MAP is None:
        npz = os.path.join(_ROOT, "map", "golan_full_dataset_cropped.npz")
        _FULL_MAP = Map(filename=npz)
    return _FULL_MAP


def crop_map(full, x0, y0, size):
    """전체 맵에서 (x0,y0) 기준 size×size 영역을 잘라 Map 인스턴스로 반환.
    Map 의 메서드(is_visible/is_passable/is_road/movement_factor 등)를 그대로 쓰기 위해
    Map.__new__ 로 만들고 필요한 배열만 슬라이스해 채운다 (cost_map 도 그대로 잘라 재사용)."""
    cm = Map.__new__(Map)
    ys, xs = slice(y0, y0 + size), slice(x0, x0 + size)
    cm.resolution_m = full.resolution_m
    cm.dem_arr = full.dem_arr[ys, xs].copy()
    cm.slope_arr = full.slope_arr[ys, xs].copy()
    cm.aspect_arr = full.aspect_arr[ys, xs].copy()
    cm.road_mask = full.road_mask[ys, xs].copy()
    cm.lake_mask = full.lake_mask[ys, xs].copy()
    cm.stream_mask = full.stream_mask[ys, xs].copy()
    cm.wood_mask = full.wood_mask[ys, xs].copy()
    cm.grid = full.grid[ys, xs].copy()
    cm.cost_map = full.cost_map[ys, xs].copy()
    cm.height, cm.width = size, size
    cm.reference_altitude = float(cm.dem_arr.min())
    cm.terrain_cost = full.terrain_cost
    cm.flow_fields = {}
    return cm


def _passable_fraction(cm):
    return float(np.mean(np.isfinite(cm.cost_map)))


def _random_passable(cm, rng, x_lo, x_hi, y_lo, y_hi, tries=300):
    """지정 영역 안에서 통행 가능한 (x, y) 셀 하나를 뽑는다. 실패 시 None."""
    for _ in range(tries):
        x = int(rng.integers(x_lo, x_hi))
        y = int(rng.integers(y_lo, y_hi))
        if cm.is_passable(x, y):
            return x, y
    return None


# 직사화력 위주의 단순 편성(간접화력/보급은 골격 단계에서 제외 — DESIGN 미정 사항)
DEFAULT_COMP = {
    "blue": ["Sho't_Kal", "Sho't_Kal", "BGM-71_TOW", "M113"],
    "red": ["T-55", "T-55", "RPG-7", "BMP-1"],
}


def spawn_episode(full, rng, size=120, comp=None):
    """랜덤 크롭 + 양 팀 유닛 스폰 + 팀별 goal 지정.
    반환: (cmap, troops, goals)  goals = {"blue": Coord, "red": Coord}
    - blue 는 좌측, red 는 우측 영역에 스폰하고 서로의 진영 쪽을 goal 로 받음.
    """
    comp = comp or DEFAULT_COMP
    # 통행 가능 비율이 너무 낮은(호수 등) 크롭은 피한다
    for _ in range(50):
        x0 = int(rng.integers(0, full.width - size))
        y0 = int(rng.integers(0, full.height - size))
        cm = crop_map(full, x0, y0, size)
        if _passable_fraction(cm) > 0.6:
            break

    troops = []

    def spawn_team(team, x_lo, x_hi):
        for name in comp[team]:
            cell = _random_passable(cm, rng, x_lo, x_hi, size // 6, size * 5 // 6)
            if cell is None:
                continue
            x, y = cell
            z = float(cm.dem_arr[y, x])
            t = Troop(name, Coord(x, y, z), affiliation=f"{team}_force", phase="RL")
            t.active = True
            t.can_move = True
            troops.append(t)

    spawn_team("blue", size // 12, size // 3)          # 좌측
    spawn_team("red", size * 2 // 3, size * 11 // 12)  # 우측

    # 팀 goal: 상대 진영 쪽 통행 가능 셀
    bg = _random_passable(cm, rng, size * 2 // 3, size * 11 // 12, size // 4, size * 3 // 4)
    rg = _random_passable(cm, rng, size // 12, size // 3, size // 4, size * 3 // 4)
    goals = {
        "blue": Coord(*(bg or (size - 1, size // 2)), 0.0),
        "red": Coord(*(rg or (0, size // 2)), 0.0),
    }
    return cm, troops, goals
