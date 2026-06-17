# map.py

import numpy as np

import math
from heapq import heappush, heappop
from typing import List, Tuple, Optional
from .unit_definitions import UnitType #, UnitStatus, UnitType, UnitComposition, HitState, UNIT_SPECS, get_landing_data, AMMUNITION_DATABASE, AmmunitionInfo, SUPPLY_DATABASE


# MAX_TIME = 100.0 # 최대 시뮬레이션 시간 (분 단위) #for testingfrom typing import List, Tuple
MAX_TIME = 2880.0  # 300 # 2880.0 # 500.0  # 최대 시뮬레이션 시간 (분 단위) 
# TIME_STEP = 0.01 # 시뮬레이션 시간 간격 (분 단위)
TIME_STEP = 1.0
# MAP_WIDTH = 30  # 맵의 너비
# MAP_HEIGHT = 30  # 맵의 높이

#!CLAUDE 성능: 8방향 이웃 (dx, dy, base_dist). 매 호출마다 리스트를 새로 만들지 않도록 모듈 상수로 분리.
_NEIGHBOR_DIRS = (
    (-1, -1, 1.15), (-1, 0, 1.0), (-1, 1, 1.15),
    (0, -1, 1.0),                  (0, 1, 1.0),
    (1, -1, 1.15),  (1, 0, 1.0),  (1, 1, 1.15),
)

class Velocity:  # Velocity class to store velocity information
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


class Coord:  # Coordinate class to store x, y, z coordinates
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    def next_coord(self, velocity: Velocity):
        # Update the coordinates based on velocity and time
        # self.x += velocity.x * TIME_STEP
        # self.y += velocity.y * TIME_STEP
        # self.z += velocity.z * TIME_STEP

        self.x += velocity.x
        self.y += velocity.y
        self.z += velocity.z

class Map:  # Map class to store map information
    def __init__(self, filename = "map/golan_full_dataset_cropped.npz"): #, width, height):
        self.resolution_m = 10
        # (1) 데이터 로드
        data = np.load(filename, allow_pickle=True)

        # 1) 래스터/마스크 레이어
        self.dem_arr       = data["dem"]
        self.aspect_arr    = data["aspect"]
        self.slope_arr     = data["slope"]
        self.road_mask     = data["road_mask"]
        self.lake_mask     = data["lake_mask"]
        self.stream_mask   = data["stream_mask"]
        self.wood_mask     = data["wood_mask"]
        
        # 3) 메타정보 복원
        transform_arr = data["transform"]            # (6,) 배열
        # transform     = Affine.from_gdal(*transform_arr)
        crs_str       = str(data["crs"].item())      # e.g. "EPSG:3857"
        # crs           = CRS.from_string(crs_str)

        self.height, self.width = self.dem_arr.shape
        self.reference_altitude = self.dem_arr[-1][0] # min([min(s) for s in self.dem_arr])
        
        # terrain_cost 맵 (필요에 따라 값 조정)
        # 0: 평지, 1: 험지, 2: 도로, 3: 호수, 4: 숲, 5: 개울
        self.terrain_cost = {
            0: 1.0,  # plain
            1: 1.3,  # rugged (slope 10)
            2: 3.5,  # rugged (slope 15)
            3: 5.0,  # rugged (slope 20)
            4: 15.0,  # rugged (slope 30)
            5: np.inf,  # rugged (slope 35)
            6: 0.8,  # road
            7: np.inf,  # lake
            8: 1.8,  # wood (forest)
            9: 2.5,  # stream (smaller water)
        }

        # 1) 빈 grid 생성 (height x width)
        self.grid = np.ones((self.height, self.width), dtype=float)

        # 2) slope 기준으로 험지 마킹 (optional)
        slope_threshold = 10.0  # degree 단위 예시 값
        self.grid[self.slope_arr > slope_threshold] *= self.terrain_cost[1]
        # 2) slope 기준으로 험지 마킹 (optional)
        slope_threshold = 15.0  # degree 단위 예시 값
        self.grid[self.slope_arr > slope_threshold] *= self.terrain_cost[2]
        slope_threshold = 20.0  # degree 단위 예시 값
        self.grid[self.slope_arr > slope_threshold] *= self.terrain_cost[3]
        slope_threshold = 30.0  # degree 단위 예시 값
        self.grid[self.slope_arr > slope_threshold] *= self.terrain_cost[4]
        slope_threshold = 35.0  # degree 단위 예시 값
        self.grid[self.slope_arr > slope_threshold] *= self.terrain_cost[5]

        # 3) 도로, 호수, 숲, 개울 덮어쓰기
        #    (마스크가 True/1인 곳에 해당 코드 적용)
        self.grid[self.road_mask.astype(bool)]   *= self.terrain_cost[6]
        self.grid[self.lake_mask.astype(bool)]   *= self.terrain_cost[7] #lask = np.inf
        self.grid[self.wood_mask.astype(bool)]   *= self.terrain_cost[8]
        self.grid[self.stream_mask.astype(bool)] *= self.terrain_cost[9]
        
        #!TEMP 비용 맵과 플로우 필드 생성 >>>>
        self.cost_map = self.build_cost_map()
        self.flow_fields = {}  # 목표별 플로우 필드 캐시
        #!TEMP 비용 맵과 플로우 필드 생성 <<<<

    #!TEMP >>>>
    def build_cost_map(self, slope_weight=0.1, min_cost=0.1):
        """각 셀의 이동 비용 계산"""
        h, w = self.slope_arr.shape
        cost_map = np.zeros((h, w), dtype=float)
        
        for i in range(h):
            for j in range(w):
                terrain_type = self.grid[i, j]
                base_cost = self.terrain_cost.get(terrain_type, 1.0)
                
                if base_cost == np.inf:
                    cost_map[i, j] = np.inf
                else:
                    # 경사도 추가 비용
                    slope = float(self.slope_arr[i, j])
                    slope_cost = 1.0 + slope * slope_weight
                    cost_map[i, j] = max(min_cost, base_cost * slope_cost)
                    
        return cost_map
    
    def is_passable(self, x: int, y: int) -> bool:
        """해당 위치가 통과 가능한지 확인"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return self.cost_map[y, x] != np.inf

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int, float]]:
        """8방향 이웃 셀과 이동 비용 반환"""
        #!CLAUDE 성능: is_passable() 호출과 매 호출 시 directions 리스트 생성을 제거(인라인화). 반환값 동일.
        # neighbors = []
        # directions = [
        #     (-1, -1, 1.15), (-1, 0, 1.0), (-1, 1, 1.15),
        #     (0, -1, 1.0),                   (0, 1, 1.0),
        #     (1, -1, 1.15),  (1, 0, 1.0),  (1, 1, 1.15)
        # ]
        #
        # for dx, dy, base_dist in directions:
        #     nx, ny = x + dx, y + dy
        #     if self.is_passable(nx, ny):
        #         cost = self.cost_map[ny, nx] * base_dist
        #         neighbors.append((nx, ny, cost))
        #
        # return neighbors
        cost_map = self.cost_map
        w, h = self.width, self.height
        neighbors = []
        for dx, dy, base_dist in _NEIGHBOR_DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                c = cost_map[ny, nx]
                if c != np.inf:
                    neighbors.append((nx, ny, c * base_dist))
        return neighbors
    #!TEMP <<<<
    
    # def is_impassable(self, x, y) -> bool:
    #     xi, yi = int(x), int(y)
    #     if not (0 <= yi < self.height and 0 <= xi < self.width):
    #         return True
    #     # 호수는 통과 불가
    #     if self.lake_mask[yi, xi]:
    #         return True
    #     return False
    
    def get_slope(self, x, y) -> float:
        """현재 좌표(x,y)의 경사(도 단위)를 반환."""
        xi, yi = int(x), int(y)
        if 0 <= yi < self.height and 0 <= xi < self.width:
            return float(self.slope_arr[yi, xi])
        return 0.0

    def get_aspect(self, x, y) -> float:
        """
        경사 방향(방위각, degree 단위: 0°=북, 90°=동, 180°=남, 270°=서)
        """
        xi, yi = int(x), int(y)
        if 0 <= yi < self.height and 0 <= xi < self.width:
            return float(self.aspect_arr[yi, xi])
        return 0.0

    def is_road(self, x, y) -> bool:
        xi, yi = int(x), int(y)
        return (0 <= yi < self.height and 0 <= xi < self.width
                and self.road_mask[yi, xi])

    def add_obstacle(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = 1  # Mark as obstacle

    def is_obstacle(self, x, y):
        return self.grid[y][x] == 1

    def movement_factor(self, x, y):
        xi, yi = int(x), int(y)
        # code = (
        #     self.grid[yi, xi] if (0 <= xi < self.width and 0 <= yi < self.height) else 0
        # )
        # # return self.terrain_cost.get(code, 1.0)
        if (0 <= xi < self.width and 0 <= yi < self.height):
            return self.grid[yi, xi]
        else:
            return 1.0    
    # def get_terrain(self, x, y):  # Get terrain type at (x, y)
    #     if 0 <= x < self.width and 0 <= y < self.height:
    #         return self.grid[x][y]
    #     return None

    # def is_road(self, x, y):
    #     xi, yi = int(x), int(y)
    #     return 0 <= xi < self.width and 0 <= yi < self.height and self.grid[xi, yi] == 2

    def is_visible(self, from_coord: Coord, to_coord: Coord, observer_height=2.0, target_height=2.0):
        """
        Fast line-of-sight check using integer grid steps.
        Returns True if there is a clear LOS from from_coord to to_coord.
        Elevations are taken from DEM; observer/target heights in meters.
        """
        x0, y0 = int(round(from_coord.x)), int(round(from_coord.y))
        x1, y1 = int(round(to_coord.x)), int(round(to_coord.y))

        if not (0 <= x0 < self.width and 0 <= y0 < self.height):
            return False
        if not (0 <= x1 < self.width and 0 <= y1 < self.height):
            return False

        z0 = self.dem_arr[y0, x0] + observer_height
        z1 = self.dem_arr[y1, x1] + target_height

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        n = max(dx, dy)
        if n == 0:
            return True  # Same cell

        #!CLAUDE 성능(M2): per-step 파이썬 루프를 numpy 벡터화. 동일한 정수 샘플점·판정식 → 결과 동일.
        #         (양 끝점이 맵 안이면 선분 내부 점도 맵 안이므로 원래의 per-step bounds 체크는 불필요.)
        # for step in range(1, n):
        #     t = step / n
        #     xi = int(round(x0 + (x1 - x0) * t))
        #     yi = int(round(y0 + (y1 - y0) * t))
        #     zi_expected = z0 + (z1 - z0) * t
        #
        #     # Bounds check (just in case)
        #     if not (0 <= xi < self.width and 0 <= yi < self.height):
        #         return False
        #
        #     ground_z = self.dem_arr[yi, xi]
        #     if ground_z + 0.5 > zi_expected:  # Blocked
        #         return False
        #
        # return True
        t = np.arange(1, n) / n
        xi = np.round(x0 + (x1 - x0) * t).astype(np.intp)
        yi = np.round(y0 + (y1 - y0) * t).astype(np.intp)
        zi_expected = z0 + (z1 - z0) * t
        ground_z = self.dem_arr[yi, xi]
        if np.any(ground_z + 0.5 > zi_expected):  # 한 점이라도 막히면 차단
            return False
        return True

        
#!TEMP >>>>
def astar_pathfinding(battle_map: Map, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """A* 알고리즘으로 최적 경로 탐색"""
    #!CLAUDE 성능: get_neighbors() 호출(노드당 list/tuple 생성, 18M+ 호출)을 인라인 전개로 대체.
    #         탐색 순서·비용·휴리스틱·heap tie-break이 모두 동일하므로 반환 경로는 동일.
    #         또한 한 번도 읽히지 않던 f_score dict 제거(불필요한 메모리/연산).
    # def heuristic(a, b):
    #     return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    #
    # open_set = []
    # heappush(open_set, (0, start))
    #
    # came_from = {}
    # g_score = {start: 0}
    # f_score = {start: heuristic(start, goal)}
    #
    # while open_set:
    #     current = heappop(open_set)[1]
    #
    #     if current == goal:
    #         # 경로 재구성
    #         path = []
    #         while current in came_from:
    #             path.append(current)
    #             current = came_from[current]
    #         path.append(start)
    #         return path[::-1]
    #
    #     for neighbor_x, neighbor_y, move_cost in battle_map.get_neighbors(*current):
    #         neighbor = (neighbor_x, neighbor_y)
    #         tentative_g = g_score[current] + move_cost
    #
    #         if neighbor not in g_score or tentative_g < g_score[neighbor]:
    #             came_from[neighbor] = current
    #             g_score[neighbor] = tentative_g
    #             f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
    #             heappush(open_set, (f_score[neighbor], neighbor))
    #
    # return []  # 경로를 찾을 수 없음
    cost_map = battle_map.cost_map
    w, h = battle_map.width, battle_map.height
    goal_x, goal_y = goal

    open_set = []
    heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    #!CLAUDE 성능: stale heap 항목 스킵용. 원본은 같은 노드를 여러 번 pop할 때마다 이웃을 재확장했는데,
    #         재확장은 항상 best g(g_score 값)를 사용하므로 no-op(이웃 갱신 없음)이다. 이를 건너뛴다.
    #         best_f[node]보다 큰 f로 pop된 항목만 스킵 → heap tuple/타이브레이크 불변, 반환 경로 동일.
    best_f = {start: 0}

    while open_set:
        f_pop, current = heappop(open_set)

        # 더 좋은 경로로 이미 처리된 항목이면 재확장 생략 (결과 불변)
        if f_pop > best_f[current]:
            continue

        if current == goal:
            # 경로 재구성
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        cx, cy = current
        cg = g_score[current]
        # get_neighbors()와 동일한 (dx, dy, base_dist) 순서 및 비용식
        for dx, dy, base_dist in _NEIGHBOR_DIRS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                c = cost_map[ny, nx]
                if c == np.inf:
                    continue
                neighbor = (nx, ny)
                tentative_g = cg + c * base_dist

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + math.sqrt((nx - goal_x) ** 2 + (ny - goal_y) ** 2)
                    best_f[neighbor] = f
                    heappush(open_set, (f, neighbor))

    return []  # 경로를 찾을 수 없음

# def build_flow_field(battle_map: Map, goal: Tuple[int, int]) -> np.ndarray:
#     """플로우 필드 생성 - 모든 셀에서 목표로의 최적 방향"""
#     h, w = battle_map.height, battle_map.width
#     flow_field = np.zeros((h, w, 2), dtype=float)
#     distance_field = np.full((h, w), np.inf)
    
#     # Dijkstra 알고리즘으로 최단 거리 계산
#     pq = []
#     goal_x, goal_y = goal
#     distance_field[goal_y, goal_x] = 0
#     heappush(pq, (0, goal_x, goal_y))
    
#     while pq:
#         dist, x, y = heappop(pq)
        
#         if dist > distance_field[y, x]:
#             continue
            
#         for nx, ny, cost in battle_map.get_neighbors(x, y):
#             new_dist = dist + cost
#             if new_dist < distance_field[ny, nx]:
#                 distance_field[ny, nx] = new_dist
#                 heappush(pq, (new_dist, nx, ny))
    
#     # 각 셀에서 가장 가까운 이웃으로의 방향 계산
#     for y in range(h):
#         for x in range(w):
#             if distance_field[y, x] == np.inf:
#                 continue
                
#             best_dir = (0, 0)
#             best_dist = distance_field[y, x]
            
#             for nx, ny, _ in battle_map.get_neighbors(x, y):
#                 if distance_field[ny, nx] < best_dist:
#                     best_dist = distance_field[ny, nx]
#                     best_dir = (nx - x, ny - y)
            
#             # 방향 벡터 정규화
#             if best_dir != (0, 0):
#                 length = math.sqrt(best_dir[0]**2 + best_dir[1]**2)
#                 flow_field[y, x] = [best_dir[0]/length, best_dir[1]/length]
    
#     return flow_field

def build_flow_field(battle_map: Map, goal: Tuple[int, int]) -> np.ndarray:

    """🟢 개선된 플로우 필드 생성"""
    h, w = battle_map.height, battle_map.width
    flow_field = np.zeros((h, w, 2), dtype=float)
    distance_field = np.full((h, w), np.inf)
    goal_x, goal_y = goal

    # Dijkstra 알고리즘으로 최단 거리 계산
    #!CLAUDE 성능: get_neighbors()의 셀당 list/tuple 생성을 인라인 전개로 대체. distance_field 결과는 동일.
    # pq = []
    # distance_field[goal_y, goal_x] = 0
    # heappush(pq, (0, goal_x, goal_y))
    #
    # while pq:
    #     dist, x, y = heappop(pq)
    #
    #     if dist > distance_field[y, x]:
    #         continue
    #
    #     for nx, ny, cost in battle_map.get_neighbors(x, y):
    #         new_dist = dist + cost
    #         if new_dist < distance_field[ny, nx]:
    #             distance_field[ny, nx] = new_dist
    #             heappush(pq, (new_dist, nx, ny))
    cost_map = battle_map.cost_map
    pq = []
    distance_field[goal_y, goal_x] = 0
    heappush(pq, (0, goal_x, goal_y))

    while pq:
        dist, x, y = heappop(pq)

        if dist > distance_field[y, x]:
            continue

        # get_neighbors()와 동일한 (dx, dy, base_dist) 및 비용식(cost_map[ny,nx]*base_dist)
        for dx, dy, base_dist in _NEIGHBOR_DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                c = cost_map[ny, nx]
                if c == np.inf:
                    continue
                new_dist = dist + c * base_dist
                if new_dist < distance_field[ny, nx]:
                    distance_field[ny, nx] = new_dist
                    heappush(pq, (new_dist, nx, ny))

    # 🟢 개선된 방향 계산 - 더 부드러운 방향 벡터
    #!CLAUDE 성능: 셀별 이중 루프(h*w * 8이웃)를 numpy 벡터화로 대체. 누적/연산 순서를 원본과 동일하게 맞춰 결과 동일.
    # for y in range(h):
    #     for x in range(w):
    #         if distance_field[y, x] == np.inf:
    #             continue
    #
    #         # 🎯 핵심 수정: 목적지로의 직선 방향을 우선 고려
    #         direct_dx = goal_x - x
    #         direct_dy = goal_y - y
    #         direct_dist = math.sqrt(direct_dx**2 + direct_dy**2)
    #
    #         if direct_dist == 0:
    #             continue
    #
    #         # 직선 방향 단위 벡터
    #         direct_ux = direct_dx / direct_dist
    #         direct_uy = direct_dy / direct_dist
    #
    #         # 🔧 그래디언트 기반 방향 계산
    #         grad_x, grad_y = 0, 0
    #         weight_sum = 0
    #
    #         # 주변 8방향의 거리 차이로 그래디언트 계산
    #         for dx in [-1, 0, 1]:
    #             for dy in [-1, 0, 1]:
    #                 if dx == 0 and dy == 0:
    #                     continue
    #
    #                 nx, ny = x + dx, y + dy
    #                 if 0 <= nx < w and 0 <= ny < h:
    #                     if distance_field[ny, nx] < distance_field[y, x]:
    #                         weight = 1.0 / max(1, abs(dx) + abs(dy))  # 대각선은 가중치 낮춤
    #                         grad_x += dx * weight
    #                         grad_y += dy * weight
    #                         weight_sum += weight
    #
    #         # 그래디언트 방향 계산
    #         if weight_sum > 0:
    #             grad_x /= weight_sum
    #             grad_y /= weight_sum
    #             grad_length = math.sqrt(grad_x**2 + grad_y**2)
    #
    #             if grad_length > 0:
    #                 grad_ux = grad_x / grad_length
    #                 grad_uy = grad_y / grad_length
    #
    #                 # 🎯 핵심 수정: 직선 방향과 그래디언트 방향을 혼합
    #                 # 직선 방향에 70% 가중치, 그래디언트 방향에 30% 가중치
    #                 final_x = direct_ux * 0.7 + grad_ux * 0.3
    #                 final_y = direct_uy * 0.7 + grad_uy * 0.3
    #
    #                 # 최종 방향 정규화
    #                 final_length = math.sqrt(final_x**2 + final_y**2)
    #                 if final_length > 0:
    #                     flow_field[y, x] = [final_x/final_length, final_y/final_length]
    #                 else:
    #                     flow_field[y, x] = [direct_ux, direct_uy]
    #             else:
    #                 # 그래디언트를 계산할 수 없으면 직선 방향 사용
    #                 flow_field[y, x] = [direct_ux, direct_uy]
    #         else:
    #             # 그래디언트를 계산할 수 없으면 직선 방향 사용
    #             flow_field[y, x] = [direct_ux, direct_uy]
    #
    # return flow_field

    # ----- 위 이중 루프의 벡터화 버전 (셀별 계산이 동일하도록 작성) -----
    YY, XX = np.mgrid[0:h, 0:w]

    # 목적지로의 직선 방향 단위 벡터 (direct_dist==0 인 목적지 셀은 0으로 보호)
    direct_dx = (goal_x - XX).astype(float)
    direct_dy = (goal_y - YY).astype(float)
    direct_dist = np.sqrt(direct_dx ** 2 + direct_dy ** 2)
    nonzero = direct_dist > 0
    safe_dd = np.where(nonzero, direct_dist, 1.0)
    direct_ux = direct_dx / safe_dd
    direct_uy = direct_dy / safe_dd

    # 주변 8방향 그래디언트 누적 — 원본 루프 순서(dx in [-1,0,1], dy in [-1,0,1])를 그대로 유지
    grad_x = np.zeros((h, w))
    grad_y = np.zeros((h, w))
    weight_sum = np.zeros((h, w))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            # neigh[y, x] = distance_field[y+dy, x+dx], 범위 밖은 inf (원본의 bounds 체크와 동일)
            neigh = np.full((h, w), np.inf)
            ys0, ys1 = max(0, -dy), min(h, h - dy)
            xs0, xs1 = max(0, -dx), min(w, w - dx)
            neigh[ys0:ys1, xs0:xs1] = distance_field[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
            mask = neigh < distance_field  # 더 가까운 이웃만 (inf 중심셀은 valid 마스크에서 제외)
            weight = 1.0 / max(1, abs(dx) + abs(dy))
            grad_x += (dx * weight) * mask
            grad_y += (dy * weight) * mask
            weight_sum += weight * mask

    # 원본의 분기 로직을 그대로 따라감
    has_w = weight_sum > 0
    safe_ws = np.where(has_w, weight_sum, 1.0)
    gx = np.where(has_w, grad_x / safe_ws, 0.0)
    gy = np.where(has_w, grad_y / safe_ws, 0.0)
    grad_length = np.sqrt(gx ** 2 + gy ** 2)
    has_g = grad_length > 0
    safe_gl = np.where(has_g, grad_length, 1.0)
    grad_ux = gx / safe_gl
    grad_uy = gy / safe_gl

    final_x = direct_ux * 0.7 + grad_ux * 0.3
    final_y = direct_uy * 0.7 + grad_uy * 0.3
    final_length = np.sqrt(final_x ** 2 + final_y ** 2)
    has_f = final_length > 0
    safe_fl = np.where(has_f, final_length, 1.0)

    # 혼합 방향을 쓰는 경우는 (weight_sum>0 & grad_length>0 & final_length>0) 뿐, 그 외에는 직선 방향
    use_mixed = has_w & has_g & has_f
    res_x = np.where(use_mixed, final_x / safe_fl, direct_ux)
    res_y = np.where(use_mixed, final_y / safe_fl, direct_uy)

    # 도달 불가(inf) 셀과 목적지 셀(direct_dist==0)은 [0, 0] 유지
    valid = np.isfinite(distance_field) & nonzero
    flow_field[:, :, 0] = np.where(valid, res_x, 0.0)
    flow_field[:, :, 1] = np.where(valid, res_y, 0.0)

    return flow_field

# 전술적 이동 패턴 추가
class TacticalManager:
    """전술적 이동 패턴 관리"""
    
    @staticmethod
    def get_tactical_destination(troop, target, battle_map: Map, allied_troops: List):
        """부대 유형과 상황에 따른 전술적 목적지 계산"""
        
        if not target:
            return troop.coord
        
        # 전차: 측면 공격 시도
        if troop.type == UnitType.TANK:
            return TacticalManager.get_flanking_position(troop, target, battle_map)
        
        # 대전차 무기: 매복 위치 선택
        elif UnitType.is_anti_tank(troop.type):
            return TacticalManager.get_ambush_position(troop, target, battle_map)
        
        # 보병: 엄폐물 활용
        elif troop.type == UnitType.INFANTRY:
            return TacticalManager.get_cover_position(troop, target, battle_map)
        
        # 간접화력: 화력지원 위치
        elif UnitType.is_indirect_fire(troop.type):
            return TacticalManager.get_fire_support_position(troop, target, battle_map)
        
        # 기본: 직접 접근
        else:
            return target.coord
    
    @staticmethod
    def get_flanking_position(troop, target, battle_map: Map):
        """측면 공격 위치 계산"""
        target_x, target_y = target.coord.x, target.coord.y
        
        # 목표 주변 90° 좌우 측면 위치들 검사
        flank_positions = []
        for angle in range(-90, 91, 30):  # -90°~90°, 30° 간격
            rad = math.radians(angle)
            
            # 500m 거리의 측면 위치
            distance = 50  # 픽셀 단위 (500m)
            flank_x = target_x + distance * math.cos(rad)
            flank_y = target_y + distance * math.sin(rad)
            
            # 지형이 통과 가능하고 유리한 위치인지 확인
            if battle_map.is_passable(int(flank_x), int(flank_y)):
                # 고도가 높은 위치 선호
                elevation = battle_map.dem_arr[int(flank_y), int(flank_x)]
                target_elevation = battle_map.dem_arr[int(target_y), int(target_x)]
                
                score = elevation - target_elevation  # 고도 차이
                
                # 숲이나 엄폐물 가까이 있으면 추가 점수
                if battle_map.wood_mask[int(flank_y), int(flank_x)]:
                    score += 10
                
                flank_positions.append((Coord(flank_x, flank_y, elevation), score))
        
        # 가장 유리한 위치 선택
        if flank_positions:
            best_pos = max(flank_positions, key=lambda x: x[1])
            return best_pos[0]
        
        return target.coord
    
    @staticmethod
    def get_ambush_position(troop, target, battle_map: Map):
        """매복 위치 계산 - 엄폐물과 사거리 고려"""
        target_x, target_y = target.coord.x, target.coord.y
        weapon_range = troop.range_km * 100  # km를 픽셀로 변환
        
        ambush_positions = []
        
        # 무기 사거리 내에서 엄폐 가능한 위치 탐색
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            
            # 사거리의 80% 거리에서 매복
            distance = weapon_range * 0.8
            amb_x = target_x + distance * math.cos(rad)
            amb_y = target_y + distance * math.sin(rad)
            
            if not battle_map.is_passable(int(amb_x), int(amb_y)):
                continue
            
            score = 0
            
            # 숲이나 높은 곳 선호
            if battle_map.wood_mask[int(amb_y), int(amb_x)]:
                score += 20
            
            elevation = battle_map.dem_arr[int(amb_y), int(amb_x)]
            target_elevation = battle_map.dem_arr[int(target_y), int(target_x)]
            if elevation > target_elevation:
                score += 15
            
            # 도로에서 멀수록 좋음 (은밀성)
            if not battle_map.road_mask[int(amb_y), int(amb_x)]:
                score += 10
            
            ambush_positions.append((Coord(amb_x, amb_y, elevation), score))
        
        if ambush_positions:
            best_pos = max(ambush_positions, key=lambda x: x[1])
            return best_pos[0]
        
        return target.coord
    
    @staticmethod
    def get_cover_position(troop, target, battle_map: Map):
        """엄폐 위치 계산"""
        # 목표와의 중간 지점에서 엄폐물 찾기
        mid_x = (troop.coord.x + target.coord.x) / 2
        mid_y = (troop.coord.y + target.coord.y) / 2
        
        cover_positions = []
        
        # 중간 지점 주변에서 엄폐 가능한 위치 탐색
        for dx in range(-20, 21, 5):
            for dy in range(-20, 21, 5):
                cover_x, cover_y = mid_x + dx, mid_y + dy
                
                if not battle_map.is_passable(int(cover_x), int(cover_y)):
                    continue
                
                score = 0
                
                # 숲, 건물, 높은 지형 선호
                if battle_map.wood_mask[int(cover_y), int(cover_x)]:
                    score += 25
                
                # 고도 차이
                elevation = battle_map.dem_arr[int(cover_y), int(cover_x)]
                if elevation > battle_map.dem_arr[int(target.coord.y), int(target.coord.x)]:
                    score += 10
                
                cover_positions.append((Coord(cover_x, cover_y, elevation), score))
        
        if cover_positions:
            best_pos = max(cover_positions, key=lambda x: x[1])
            return best_pos[0]
        
        return target.coord
    
    @staticmethod
    def get_fire_support_position(troop, target, battle_map: Map):
        """화력지원 위치 계산"""
        # 간접화력은 목표에서 멀리, 높은 곳에서 사격
        target_x, target_y = target.coord.x, target.coord.y
        weapon_range = troop.range_km * 100  # km를 픽셀로 변환
        
        support_positions = []
        
        # 사거리 내에서 가장 높은 위치 찾기
        for distance in [weapon_range * 0.7, weapon_range * 0.8, weapon_range * 0.9]:
            for angle in range(0, 360, 30):
                rad = math.radians(angle)
                sup_x = target_x + distance * math.cos(rad)
                sup_y = target_y + distance * math.sin(rad)
                
                if not battle_map.is_passable(int(sup_x), int(sup_y)):
                    continue
                
                elevation = battle_map.dem_arr[int(sup_y), int(sup_x)]
                
                # 고도가 높을수록, 도로 접근성이 좋을수록 선호
                score = elevation
                if battle_map.road_mask[int(sup_y), int(sup_x)]:
                    score += 20
                
                support_positions.append((Coord(sup_x, sup_y, elevation), score))
        
        if support_positions:
            best_pos = max(support_positions, key=lambda x: x[1])
            return best_pos[0]
        
        return target.coord
#!TEMP <<<<