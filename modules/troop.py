# troop.py

import math
import numpy as np
import random
from .map import Coord, Velocity, Map, MAX_TIME, TIME_STEP
from .unit_definitions import UnitStatus, UnitType, UnitComposition, HitState, UNIT_SPECS, get_landing_data, AMMUNITION_DATABASE, AmmunitionInfo, SUPPLY_DATABASE

#!TEMP >>>>
from .map import astar_pathfinding, TacticalManager, build_flow_field
from typing import List, Tuple, Optional
#!TEMP <<<<

BLUE_RANGE_BUFF = 1.2  # BLUE 진영의 사거리 버프 (20% 증가)
BLUE_HIT_PROB_BUFF = 0.8  # BLUE 진영의 명중 확률 버프
BLUE_OBS_BUFF = 1.2  # BLUE 진영의 관측 버프 (1.5배 더 관측 가능)
# RED_RANGE_BUFF = 0.8  # RED 진영의 사거리 버프 (기본값, 0% 증가)
PATHFIND_COOLDOWN = 30.0  # A* 경로 재계산 쿨타임 (초 단위)
#!CLAUDE 메모리: 플로우 필드 1개 ≈ 8MB(649*791*2 float64). 무한 캐시 증가를 막기 위한 상한(FIFO 제거).
#         캐시 미스 시 재계산하며 결과는 동일하므로 시뮬레이션 동작은 변하지 않음.
FLOW_FIELD_CACHE_MAX = 32

class Troop:  # Troop class to store troop information and actions
    # Static variables to keep track of troop IDs
    counter = {}

    def __init__(self, unit_name, coord=Coord(), affiliation: str = None, phase: str = None, fixed_dest=None):
        spec = UNIT_SPECS[unit_name]
        self.spec = spec
        self.team = spec.team
        self.type = spec.unit_type
        self.name = spec.name
        self.range_km = spec.range_km
        if self.team == "blue":
            self.range_km *= BLUE_RANGE_BUFF  # Apply BLUE team range buff
        self.ph_func = spec.ph_func
        self.pk_func = spec.pk_func
        self.damage_func = spec.damage_func
        self.target_delay_func = spec.target_delay_func
        self.fire_time_func = spec.fire_time_func
        self.active   = False   # 이벤트상 “활성” 여부 (가시/표적 대상 등)
        self.can_move = False   # 이벤트상 “이동 허용” 여부
        self.observed = False  # 이벤트상 “관측 대상” 여부
        # self.firing = False  # 이벤트상 “사격 중” 여부

        self.id = self.assign_id()
        self.next_fire_time = 0.0  # Initial fire time
        self.target = None
        self.target_coord = None
        self.alive = True
        self.coord = coord  # Coordinate object to store (x, y, z) coordinates
        self.velocity = Velocity()  # Placeholder for velocity (x, y, z)
        self.status = UnitStatus.ALIVE  # Placeholder for unit status

        self.affiliation = affiliation
        self.phase = phase
        self.fixed_dest = fixed_dest  # Fixed dest, Coordinate object to store (x, y, z) coordinates

        # self.ammo = 100  # ammo level (0-100%)
        # self.supply = 100  # supply level (0-100%)
        # ammo_info = AMMUNITION_DATABASE.get(unit_name, AmmunitionInfo(0, 0, 0, 0))
        # # 적재량
        # self.main_ammo = float(ammo_info.main_ammo)
        # self.secondary_ammo = float(ammo_info.secondary_ammo)

        # # ▶ 분당 사용 속도 (= 하루 예상 사용량 ÷ 1440 분)
        # self.main_rate  = ammo_info.daily_main_usage / 1440.0
        # self.sec_rate   = ammo_info.daily_sec_usage  / 1440.0

        # self.last_ammo_check = 0.0          # 마지막 소모 계산 시각
        # self.ammo_restricted_until = 0.0    # 10 % 이하 → 5 분 금지용

        # if self.type == UnitType.SUPPLY:
        #     self.supply_stock = {}  # 예: {"T-55": 129, "AK-47": 12000, ...}
        #     for k, v in SUPPLY_DATABASE.items():
        #         self.supply_stock[k] = float(v)

        #!TEMP >>>>
        self.path = []  # A* 경로
        self.path_index = 0
        self.last_pathfind_time = 0
        #!TEMP <<<<

    def dead(self):
        del self  # Explicitly delete the object

    def assign_id(self):
        key = f"{self.team}_{self.type.value}"
        if key not in Troop.counter:
            Troop.counter[key] = 1
        label = (
            f"{self.team[0].upper()}_{self.type.value[:2].upper()}{Troop.counter[key]}"
        )
        Troop.counter[key] += 1
        return label

    def update_coord(self):  # Update coordinates
        self.coord.next_coord(self.velocity)

    def update_velocity(self, new_velocity):  # Update velocity
        self.velocity = new_velocity

    def get_distance(self, other_troop):  # Calculate distance to another troop
        return math.sqrt(
            (self.coord.x - other_troop.coord.x) ** 2
            + (self.coord.y - other_troop.coord.y) ** 2
            + (self.coord.z - other_troop.coord.z) ** 2
        ) * 0.01 # pixel -> km 변환 (10m = 0.01km)

    # def get_distance(self, other_troop):
    #     """최적화된 3D 거리 계산 - 권장 버전"""
    #     dx = (self.coord.x - other_troop.coord.x) * 10
    #     dy = (self.coord.y - other_troop.coord.y) * 10
    #     dz = self.coord.z - other_troop.coord.z
    #     return (dx*dx + dy*dy + dz*dz) ** 0.5 * 0.001

    def get_distance_fast(self, other_troop):
        # 제곱근 연산 제거한 버전
        dx = self.coord.x - other_troop.coord.x
        dy = self.coord.y - other_troop.coord.y
        return ((dx*dx + dy*dy) ** 0.5) * 0.01  # ** 0.5가 sqrt()보다 빠름

    def get_t_a(self):
        return self.target_delay_func(0) if self.target else 0

    def get_t_f(self):
        return self.fire_time_func(0) if self.target else 0

        # def compute_velocity(
        #     self, dest: Coord, battle_map: Map, current_time: float
        # ) -> Velocity: # TODO: stop if in range, hour/minute check

        #     # 1) 기본 속도 km/h → km/min
        #     on_road = battle_map.is_road(self.coord.x, self.coord.y)
        #     base_speed = (
        #         self.spec.speed_road_kmh if on_road else self.spec.speed_offroad_kmh
        #     ) / 60

        #     # 2) 지형 가중치
        #     terrain_factor = battle_map.movement_factor(self.coord.x, self.coord.y)

        #     if not np.isfinite(terrain_factor):
        #         # impassable cell → 움직이지 않음 #TODO 방향을 돌려서 가도록 전환 필요.
        #         return Velocity(0,0,0)

        #     # 3) 낮/밤 가중치 (19:00–06:00 야간엔 50% 느려짐)
        #     hour = int((13 * 60 + 55 + current_time) // 60) % 24
        #     daynight = 1.0 if 6 <= hour < 19 else 1.5

        #     # 4) 실제 per-min 이동량
        #     speed = base_speed / terrain_factor / daynight

        #     # 5) 방향 단위 벡터
        #     dx, dy = dest.x - self.coord.x, dest.y - self.coord.y
        #     dist = math.hypot(dx, dy)

        #     if dist == 0:
        #         return Velocity(0, 0, 0)

        if dist < self.range_km:  #TODO: 사거리 제한
            # 목표 지점이 사거리 이내면 멈춤
            return Velocity(0, 0, 0)

        ux, uy = dx / dist, dy / dist

    #     move = speed * TIME_STEP

    #     # move (km) → move_m (m)
    #     move_m  = move * 1000

    #     # print(f"[{self.id}] move = {move_m:.1f} m/min ("f"{speed:.3f} km/min)")

    #     # move_m (m) → move_px (pixels), given 1 px = 10 m
    #     move_px = move_m / battle_map.resolution_m

    #     # 로그 출력
    #     # direction unit vector stays the same:
    #     ux, uy = dx/dist, dy/dist

    #     # now return pixel‐per‐step velocity instead of km‐per‐step
    #     # return Velocity(ux * move_px, uy * move_px, 0) # 방향 탐색 안하면 아래 코멘트 처리 후 여기서 그만하기.
    #     return Velocity(ux * move, uy * move, 0)

    # ---- 역할별 전략: 유형별 타겟팅 로직 ----
    def filter_priority(self, cand_list): # TODO: unit type 간소화 가능
        if self.type == UnitType.TANK:
            return sorted(
                cand_list,
                key=lambda c: (
                    c[0].type != UnitType.TANK,
                    c[0].type != UnitType.ATGM,
                    c[0].type != UnitType.APC,
                    c[1],  # distance
                ),
            )
        elif self.type == UnitType.APC:
            return sorted(
                cand_list,
                key=lambda c: (
                    c[0].type != UnitType.INFANTRY,
                    c[1],
                ),
            )
        elif UnitType.is_anti_tank(self.type):
            at_targets = [
                c for c in cand_list if c[0].type in {UnitType.TANK, UnitType.APC}
            ]
            return sorted(at_targets, key=lambda c: (c[2], c[1]))
        elif UnitType.is_indirect_fire(self.type):
            return sorted(
                cand_list,
                key=lambda c: (
                    c[0].status != UnitStatus.STATIONARY,
                    c[1],
                ),
            )
        elif self.type == UnitType.INFANTRY:
            return sorted(
                cand_list,
                key=lambda c: (
                    c[0].type != UnitType.INFANTRY,
                    c[1],
                ),
            )
        elif self.type == UnitType.SUPPLY:
            self.target = None
            self.next_fire_time = float("inf")
            return
        else:
            return sorted(cand_list, key=lambda c: (c[2], c[1]))

    def find_observed_enemies(self, troop_list, battle_map: Map):
        """🟢 관측 가능한 적군 필터링"""
        if self.team == "blue":
            # 블루팀은 1.5배 관측 가능
            range_km = self.range_km * BLUE_OBS_BUFF
            observed_enemies = troop_list.red_observed
            enemies = troop_list.red_troops
        if self.team == "red":
            # 레드팀은 기본 관측 범위
            range_km = self.range_km
            observed_enemies = troop_list.blue_observed
            enemies = troop_list.blue_troops

        for troop in enemies:
            if troop in observed_enemies:
                if troop.active == False or troop.alive == False:
                    observed_enemies.remove(troop)
                else:
                    continue  # 이미 관측된 적은 무시
            else:
                # 관측 범위 내 적군만 필터링
                if troop.active and troop.alive:
                    distance = self.get_distance(troop)
                    if distance <= range_km:
                        if battle_map.is_visible(self.coord, troop.coord):
                            observed_enemies.append(troop)
        return

    def assign_target(
        self, current_time, enemy_list
    ):  # TODO: Implement target assignment logic, indirect fire logic

        #!TEMP 이미 좋은 타겟이 있으면 그대로 유지 >>>>
        if (self.target and self.target.alive and 
            getattr(self.target, 'active', False)):
            distance = self.get_distance(self.target)
            if distance <= self.range_km:
                return  # 기존 타겟 유지 - 계산 생략!        
        #!TEMP 이미 좋은 타겟이 있으면 그대로 유지 <<<<

        # 밤 시간대: 19:00 ~ 06:00
        is_night = 360 <= current_time % 1440 <= 1080

        if enemy_list:
            # 거리 계산 포함 유효 후보 필터링
            candidates = []
            for e in enemy_list:
                if not e.alive:
                    continue
                if e.status == UnitStatus.HIDDEN:
                    continue

                #!TEMP 추가: active=False인 적은 타겟 대상에서 제외
                if not getattr(e, 'active', False):
                    continue

                distance = self.get_distance(e)

                if distance > self.range_km:  #TODO: 사거리 제한
                    continue

                candidates.append((e, distance, 1))

            if not candidates:
                self.target = None
                self.next_fire_time = float("inf")
                return

            priority_list = self.filter_priority(candidates)
            if not priority_list:
                self.target = None
                self.next_fire_time = float("inf")
                return

            # 최종 타겟 지정
            best_target = priority_list[0][0]
            ta = self.get_t_a()
            if is_night:
                ta *= 1.5  # 야간 표적 획득 시간 증가

            self.target = best_target
            self.next_fire_time = round(current_time + ta + self.get_t_f(), 2)

        else:
            self.target = None
            self.next_fire_time = float("inf")
            # print("no more enemy left")
            return

    # ▶ 경과 시간만큼 탄약을 자동 소모
    def consume_ammo(self, current_time):
        dt = current_time - self.last_ammo_check
        if dt <= 0:
            return

        # 저장해두고
        prev_main = int(self.main_ammo)
        prev_sec = int(self.secondary_ammo)

        # 감소
        self.main_ammo = max(0.0, self.main_ammo - self.main_rate * dt)
        self.secondary_ammo = max(0.0, self.secondary_ammo - self.sec_rate * dt)

        # 현재 정수 단위
        curr_main = int(self.main_ammo)
        curr_sec = int(self.secondary_ammo)

        # 정수 단위 탄약 감소 여부 저장
        self.main_ammo_int_changed = (curr_main < prev_main)
        self.secondary_ammo_int_changed = (curr_sec < prev_sec)

        self.last_ammo_check = current_time

    def fire(self, current_time, enemy_list, troop_list, history):  # TODO: Implement firing logic
        # TODO 에러 발생
        # # ▶ 경과 분만큼 탄약 소모
        # self.consume_ammo(current_time)

        # total_ammo = self.main_ammo + self.secondary_ammo

        # # 10 % 이하 → 5 분 금지
        # if total_ammo <= 0.1 * (
        #      AMMUNITION_DATABASE[self.name].main_ammo +
        #      AMMUNITION_DATABASE[self.name].secondary_ammo):
        #     self.ammo_restricted_until = current_time + 5.0

        # # ▶ 탄약 없거나 제한 시간이면 발사 불가
        # if total_ammo <= 0 or current_time < self.ammo_restricted_until:
        #     self.next_fire_time = round(current_time + 1.0, 2)  # 1 분 뒤 재시도
        #     return

        if not self.alive:
            return

        #!TEMP 수정: None 체크를 먼저 수행 >>>>
        if self.target is None or not self.target.alive:
            self.assign_target(current_time, enemy_list)
            return
        #!TEMP 수정: None 체크를 먼저 수행 <<<<

        if self.target.alive == False or self.target is None:
            self.assign_target(current_time, enemy_list)
            return

        if self.status == UnitStatus.DAMAGED_FIREPOWER:
            self.next_fire_time = round(current_time + self.get_t_f(), 2)
            return

        distance = self.get_distance(self.target)

        if distance > self.range_km:  #TODO: 사거리 제한
            # self.next_fire_time = round(current_time + self.get_t_f(), 2)
            self.assign_target(current_time, enemy_list)
            return
        result = HitState.MISS
        hit_rand_var = np.random.rand()
        kill_rand_var = np.random.rand()

        if UnitType.is_indirect_fire(self.type):
            landing_x, landing_y, lethal_r = get_landing_data(self.name, self.target.coord, distance)
            landing_dist = math.sqrt(
                (self.target.coord.x - landing_x) ** 2
                + (self.target.coord.y - landing_y) ** 2
            )
            pk = self.damage_func(landing_dist, lethal_r)
            result = self.pk_func(kill_rand_var, pk)

        else:
            # if self.type in UnitType.DIR_FIRE_UNIT:
            ph = self.ph_func(self.get_distance(self.target))
            # 야간 명중률 보정 (19:00 ~ 06:00 = 360~1080분)
            if 360 <= current_time % 1440 <= 1080:
                ph *= 0.8  # 20% 감소

            if self.target.team == "blue":
                ph = ph * BLUE_HIT_PROB_BUFF
            if ph > hit_rand_var:
                result = self.pk_func(kill_rand_var)
            else:
                result = HitState.MISS

        # elif self.type in UnitType.INDIRECT_FIRE_UNIT:
        #     ph = self.ph_func(self.get_distance(self.target))
        #     if ph < hit_rand_var:
        #         result = self.pk_func(kill_rand_var)
        #     else:
        #         result = HitState.MISS

        print("Result", self.id, result, self.team, self.type, self.name, "->", self.target.id, self.target.team, self.target.name)

        troop_list.add_observed_troop(self)  # 관측 대상 추가

        history.add_to_battle_log(
            type_=self.type.value,
            shooter=self.id,
            target=self.target.id if self.target else None,
            target_type=self.target.type.value if self.target else None,
            result=result.value,
        )

        if self.target.type == UnitType.TANK:
            if result == HitState.CKILL:
                self.target.alive = False
                self.target = None
                self.assign_target(current_time, enemy_list)
                return
            elif result == HitState.MKILL:
                if self.target.status == UnitStatus.DAMAGED_FIREPOWER:
                    self.target.alive = False
                    self.target.status = UnitStatus.DESTROYED
                else:
                    self.target.status = UnitStatus.DAMAGED_MOBILITY
            elif result == HitState.FKILL:
                if self.target.status == UnitStatus.DAMAGED_MOBILITY:
                    self.target.alive = False
                    self.target.status = UnitStatus.DESTROYED
                else:
                    self.target.status = UnitStatus.DAMAGED_FIREPOWER

            self.next_fire_time = round(current_time + self.get_t_f(), 2)

        else:
            if result == HitState.MISS:
                self.next_fire_time = round(current_time + self.get_t_f(), 2)
                return
            else:
                self.target.alive = False
                self.target.status = UnitStatus.DESTROYED
                self.target = None
                self.assign_target(current_time, enemy_list)
                return

    def compute_velocity_advanced(self, dest, battle_map: Map, current_time: float):
        """🟢 개선된 이동 계산 - 직선 통과 없이"""

        # 1. 경로 재계산 조건 확인
        should_recalculate = (
            not self.path
            or current_time - self.last_pathfind_time > PATHFIND_COOLDOWN
            or self.path_index >= len(self.path)
        )

        if should_recalculate:
            self.recalculate_path(dest, battle_map, current_time)

        # 2. 경로가 있으면 경로 따라가기
        if self.path and self.path_index < len(self.path):
            return self.follow_path(battle_map, current_time)

        # 3. 경로가 없으면 직선 이동 (백업) - 장애물 회피 포함
        return self.compute_direct_velocity(dest, battle_map, current_time)

    # def recalculate_path(self, dest, battle_map: Map, current_time: float):

    #     """A* 또는 플로우 필드로 경로 재계산"""
    #     start = (int(self.coord.x), int(self.coord.y))
    #     goal = (int(dest.x), int(dest.y))

    #     # 목표가 너무 가까우면 직선 이동
    #     if math.hypot(goal[0] - start[0], goal[1] - start[1]) < 5:
    #         self.path = []
    #         return

    #     # 플로우 필드 사용 (대규모 부대용)
    #     if self.should_use_flow_field(battle_map):
    #         self.path = self.get_flow_field_path(goal, battle_map)
    #     else:
    #         # A* 사용 (개별 부대용)
    #         self.path = astar_pathfinding(battle_map, start, goal)

    #     self.path_index = 0
    #     self.last_pathfind_time = current_time

    def recalculate_path(self, dest, battle_map: Map, current_time: float):

        """🟢 개선된 경로 재계산 (직선 통과 체크 제거)"""
        start = (int(self.coord.x), int(self.coord.y))
        goal = (int(dest.x), int(dest.y))

        # 🟢 아주 가까운 거리만 경로탐색 생략
        distance_to_goal = math.hypot(goal[0] - start[0], goal[1] - start[1])
        if distance_to_goal < 3:  # 30m 이내만 생략
            self.path = []
            return

        # 🟢 항상 경로탐색 사용 (직선 통과 체크 제거)
        if self.should_use_flow_field(battle_map):
            self.path = self.get_flow_field_path(goal, battle_map)
        else:
            self.path = astar_pathfinding(battle_map, start, goal)

        # 🟢 경로 후처리: 너무 가까운 웨이포인트 제거
        if self.path:
            self.path = self.filter_close_waypoints(self.path, min_distance=3)

        self.path_index = 0
        self.last_pathfind_time = current_time

    def should_use_flow_field(self, battle_map: Map) -> bool:
        """플로우 필드 사용 여부 결정"""
        # 같은 목표를 가진 아군이 많으면 플로우 필드 사용
        # 여기서는 단순화해서 전차나 대규모 부대만 사용
        return self.type in [UnitType.TANK, UnitType.APC]

    def filter_close_waypoints(self, path: List[Tuple[int, int]], 
                             min_distance: float = 3) -> List[Tuple[int, int]]:
        """🟢 너무 가까운 웨이포인트 제거로 부드러운 경로 생성"""
        if len(path) <= 2:
            return path

        filtered = [path[0]]

        for i in range(1, len(path)):
            last_point = filtered[-1]
            current_point = path[i]

            distance = math.hypot(
                current_point[0] - last_point[0],
                current_point[1] - last_point[1]
            )

            # 최소 거리 이상일 때만 추가
            if distance >= min_distance:
                filtered.append(current_point)

        # 마지막 점은 항상 포함
        if filtered[-1] != path[-1]:
            filtered.append(path[-1])

        return filtered

    def get_flow_field_path(self, goal: Tuple[int, int], battle_map: Map) -> List[Tuple[int, int]]:
        """🟢 수정된 플로우 필드 경로 생성 - 실수 좌표 사용"""
        goal_key = f"{goal[0]}_{goal[1]}"

        if goal_key not in battle_map.flow_fields:
            #!CLAUDE 메모리: 캐시가 상한에 도달하면 가장 오래된 항목부터 제거(dict는 삽입 순서 유지).
            if len(battle_map.flow_fields) >= FLOW_FIELD_CACHE_MAX:
                oldest_key = next(iter(battle_map.flow_fields))
                del battle_map.flow_fields[oldest_key]
            battle_map.flow_fields[goal_key] = build_flow_field(battle_map, goal)

        flow_field = battle_map.flow_fields[goal_key]

        # 🟢 핵심 수정: 실수 좌표로 경로 생성
        path = []
        x, y = float(self.coord.x), float(self.coord.y)

        for step in range(200):  # 더 많은 스텝 허용
            # 목표 근처 도달 확인
            if abs(x - goal[0]) < 2 and abs(y - goal[1]) < 2:
                break

            xi, yi = int(x), int(y)
            if not (0 <= yi < battle_map.height and 0 <= xi < battle_map.width):
                break

            dx, dy = flow_field[yi, xi]
            if dx == 0 and dy == 0:
                break

            # 🟢 핵심 수정: 실수 좌표로 이동하되 작은 스텝 사용
            step_size = 2.0  # 작은 스텝으로 부드러운 곡선
            x += dx * step_size
            y += dy * step_size

            # 🟢 더 조밀한 웨이포인트 생성 (2스텝마다)
            if step % 2 == 0:
                path.append((int(x), int(y)))

        return path

    def follow_path(self, battle_map: Map, current_time: float):
        """🟢 개선된 경로 따라가기 - 부드러운 곡선 이동"""
        #!CLAUDE 안정성: 가까운 웨이포인트가 연속되면 재귀 깊이가 깊어져 스택 오버플로/세그폴트 위험.
        #         동작은 동일하게 유지하면서 재귀를 while 루프로 변환(재귀 호출 → continue).
        # if self.path_index >= len(self.path):
        #     return Velocity(0, 0, 0)
        #
        # # 🟢 현재 타겟과 다음 타겟을 고려한 스무스 이동
        # target_x, target_y = self.path[self.path_index]
        #
        # # 다음 웨이포인트가 있으면 미리보기로 방향 조정
        # if self.path_index + 1 < len(self.path):
        #     next_x, next_y = self.path[self.path_index + 1]
        #
        #     # 현재 타겟까지의 거리
        #     dx1 = target_x - self.coord.x
        #     dy1 = target_y - self.coord.y
        #     dist1 = math.hypot(dx1, dy1)
        #
        #     # 🟢 타겟에 가까워지면 다음 타겟 방향도 고려
        #     if dist1 < 8:  # 80m 이내
        #         dx2 = next_x - target_x
        #         dy2 = next_y - target_y
        #
        #         # 가중 평균으로 방향 스무딩
        #         weight = max(0, (8 - dist1) / 8)  # 가까울수록 다음 방향 가중치 증가
        #
        #         dx = dx1 * (1 - weight) + dx2 * weight
        #         dy = dy1 * (1 - weight) + dy2 * weight
        #     else:
        #         dx, dy = dx1, dy1
        # else:
        #     dx = target_x - self.coord.x
        #     dy = target_y - self.coord.y
        #
        # dist = math.hypot(dx, dy)
        #
        # # 🟢 도달 임계값 축소 (1.5픽셀 = 15m)
        # if dist < 1.5:
        #     self.path_index += 1
        #     if self.path_index >= len(self.path):
        #         return Velocity(0, 0, 0)
        #     return self.follow_path(battle_map, current_time)
        #
        # # 방향과 속도 계산
        # ux = dx / dist
        # uy = dy / dist
        #
        # move_m = self.calculate_movement_distance(battle_map, current_time)
        # move_px = move_m / battle_map.resolution_m
        #
        # # 🟢 오버슈팅 방지
        # move_px = min(move_px, dist * 0.8)  # 거리의 80%로 제한
        #
        # return Velocity(ux * move_px, uy * move_px, 0)

        # ----- 위 재귀 버전과 동작이 동일한 반복문 버전 -----
        while True:
            if self.path_index >= len(self.path):
                return Velocity(0, 0, 0)

            # 🟢 현재 타겟과 다음 타겟을 고려한 스무스 이동
            target_x, target_y = self.path[self.path_index]

            # 다음 웨이포인트가 있으면 미리보기로 방향 조정
            if self.path_index + 1 < len(self.path):
                next_x, next_y = self.path[self.path_index + 1]

                # 현재 타겟까지의 거리
                dx1 = target_x - self.coord.x
                dy1 = target_y - self.coord.y
                dist1 = math.hypot(dx1, dy1)

                # 🟢 타겟에 가까워지면 다음 타겟 방향도 고려
                if dist1 < 8:  # 80m 이내
                    dx2 = next_x - target_x
                    dy2 = next_y - target_y

                    # 가중 평균으로 방향 스무딩
                    weight = max(0, (8 - dist1) / 8)  # 가까울수록 다음 방향 가중치 증가

                    dx = dx1 * (1 - weight) + dx2 * weight
                    dy = dy1 * (1 - weight) + dy2 * weight
                else:
                    dx, dy = dx1, dy1
            else:
                dx = target_x - self.coord.x
                dy = target_y - self.coord.y

            dist = math.hypot(dx, dy)

            # 🟢 도달 임계값 축소 (1.5픽셀 = 15m)
            if dist < 1.5:
                self.path_index += 1
                if self.path_index >= len(self.path):
                    return Velocity(0, 0, 0)
                continue  # 다음 웨이포인트로 (원본의 재귀 호출과 동일)

            # 방향과 속도 계산
            ux = dx / dist
            uy = dy / dist

            move_m = self.calculate_movement_distance(battle_map, current_time)
            move_px = move_m / battle_map.resolution_m

            # 🟢 오버슈팅 방지
            move_px = min(move_px, dist * 0.8)  # 거리의 80%로 제한

            return Velocity(ux * move_px, uy * move_px, 0)

    def calculate_movement_distance(self, battle_map: Map, current_time: float) -> float:
        """이동 거리 계산 (기존 로직을 메소드로 분리)"""
        # 1) 기본 속도 km/h → km/min
        on_road = battle_map.is_road(self.coord.x, self.coord.y)
        base_speed = (
            self.spec.speed_road_kmh if on_road else self.spec.speed_offroad_kmh
        ) / 60  # km/h -> km/min

        # 2) 지형 가중치
        terrain_factor = battle_map.movement_factor(self.coord.x, self.coord.y)
        if not np.isfinite(terrain_factor):
            return 0

        # 3) 낮/밤 가중치 (19:00–06:00 야간엔 50% 느려짐)
        hour = int((13 * 60 + 55 + current_time) // 60) % 24
        daynight = 1.0 if 6 <= hour < 19 else 1.5

        # 4) 실제 per-min 이동량
        speed = base_speed / terrain_factor / daynight
        # if self.firing:
        #     speed *= 0.2
        move_km = speed * TIME_STEP  # TIME_STEP은 1.0분

        # 5) km → m → pixels 변환
        move_m = move_km * 1000  
        move_px = move_m / battle_map.resolution_m  # 10m = 1px

        return move_px

    def compute_direct_velocity(self, dest, battle_map: Map, current_time: float):
        """직선 이동 (백업용) - 장애물 회피 포함"""
        dx, dy = dest.x - self.coord.x, dest.y - self.coord.y
        dist = math.hypot(dx, dy)

        if dist == 0:
            return Velocity(0, 0, 0)

        ux, uy = dx / dist, dy / dist

        # 장애물 회피: 여러 방향 시도
        directions = self.get_avoidance_directions(ux, uy)

        # 프레임당 이동 거리 계산 (예: pixel 단위)
        raw_move = self.calculate_movement_distance(battle_map, current_time)
        # 남은 거리보다 이동량이 크면, dist로 제한
        move_distance = min(raw_move, dist)

        for dir_x, dir_y in directions:
            test_x = self.coord.x + dir_x * 3  # 3픽셀 앞 확인
            test_y = self.coord.y + dir_y * 3

            if battle_map.is_passable(int(test_x), int(test_y)):
                move_distance = self.calculate_movement_distance(battle_map, current_time)
                return Velocity(dir_x * move_distance, dir_y * move_distance, 0)

        # 모든 방향이 막혔으면 정지
        return Velocity(0, 0, 0)

    def get_avoidance_directions(self, ux: float, uy: float) -> List[Tuple[float, float]]:
        """장애물 회피를 위한 후보 방향들"""
        directions = [(ux, uy)]  # 원래 방향이 최우선

        # ±15°, ±30°, ±45° 방향 추가
        for angle in [15, 30, 45, -15, -30, -45]:
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            new_x = ux * cos_a - uy * sin_a
            new_y = ux * sin_a + uy * cos_a
            directions.append((new_x, new_y))

        return directions

    # def compute_velocity(self, dest, battle_map: Map, current_time: float):
    #     """개선된 이동 로직 - 전술적 목적지와 고급 경로탐색"""

    #     # 1. 전술적 목적지 결정
    #     if self.target and self.active and self.can_move:
    #         # 아군 부대 리스트 필요 (실제 구현시 TroopList에서 전달)
    #         allied_troops = []  # 임시
    #         tactical_dest = TacticalManager.get_tactical_destination(
    #             self, self.target, battle_map, allied_troops
    #         )
    #     else:
    #         tactical_dest = dest

    #     # 2. 고급 경로탐색 사용
    #     return self.compute_velocity_advanced(tactical_dest, battle_map, current_time)

    #!TEMP <<<<


class TroopList:  # Troop list to manage all troops
    def __init__(self, troop_list):
        self.troops = []
        self.blue_troops = []
        self.red_troops = []
        self.troop_ids = []
        self.blue_observed = []
        self.red_observed = []

        for troop in troop_list:
            self.troops.append(troop)
            if troop.id in self.troop_ids:
                print(f"[ERROR] Duplicate id: {troop.id}")
                continue
            else:
                self.troop_ids.append(troop.id)            
            if troop.team == "blue":
                self.blue_troops.append(troop)
            elif troop.team == "red":
                self.red_troops.append(troop)
        self.assign_targets(0.0)

    def remove_troop(self, troop: Troop):
        if troop in self.troops:
            self.troops.remove(troop)
            if troop.team == "blue":
                self.blue_troops.remove(troop)
                if troop in self.blue_observed:
                    self.blue_observed.remove(troop)
            elif troop.team == "red":
                self.red_troops.remove(troop)
                if troop in self.red_observed:
                    self.red_observed.remove(troop)

    def remove_dead_troops(self):
        for troop in self.troops:
            if not troop.alive:
                self.remove_troop(troop)
                troop.dead()

    def get_observed_enemies(self, team):
        if team == "blue":
            return self.red_observed
        elif team == "red":
            return self.blue_observed
        else:
            print(f"[ERROR] wrong team affiliation: {team}")
            return None

    def add_observed_troop(self, troop):
        """🟢 관측 가능한 적군 추가"""
        if troop.active and troop.alive:
            if troop.team == "blue":
                if troop not in self.blue_observed:
                    self.blue_observed.append(troop)
            elif troop.team == "red":
                if troop not in self.red_observed:
                    self.red_observed.append(troop)
            else:
                print(f"[ERROR] wrong team affiliation: {troop.team}")

    def update_observation(self, battle_map: Map):
        """🟢 관측 가능한 적군 업데이트"""
        for troop in self.troops:
            if troop.alive and getattr(troop, 'active', False):
                troop.find_observed_enemies(self, battle_map)

    def assign_targets(self, current_time):

        # ------------------
        # ========== 수정 1: 활성화된 적 리스트를 미리 필터링 ==========
        # active_blue_troops = [t for t in self.blue_troops if getattr(t, 'active', False) and t.alive]
        # active_red_troops = [t for t in self.red_troops if getattr(t, 'active', False) and t.alive]
        active_blue_troops = self.blue_observed
        active_red_troops = self.red_observed

        # ========== 수정 2: 타겟 할당 전 기존 타겟 검증 ==========
        # 죽었거나 비활성화된 타겟들을 먼저 제거
        for troop in self.troops:
            if troop.alive and troop.target:
                if (not troop.target.alive or 
                    not getattr(troop.target, 'active', False)):
                    troop.target = None
                    print(f"  {troop.id}: 기존 타겟 무효화됨")

        # ========== 수정 3: 활성화된 부대만 타겟 할당 ==========
        targets_assigned = 0
        for troop in self.troops:
            if troop.alive and getattr(troop, 'active', False):
                old_target = troop.target

                if troop.team == "blue":
                    troop.assign_target(current_time, active_red_troops)
                elif troop.team == "red":
                    troop.assign_target(current_time, active_blue_troops)

                # 새 타겟이 할당되었는지 확인
                if troop.target != old_target and troop.target is not None:
                    targets_assigned += 1
                    print(f"  {troop.id}: 새 타겟 할당 -> {troop.target.id}")

        print(f"  총 {targets_assigned}개 유닛에 새 타겟 할당됨")

        # ========== 수정 4: 타겟 할당 통계 출력 ==========
        blue_with_targets = len([t for t in active_blue_troops if t.target])
        red_with_targets = len([t for t in active_red_troops if t.target])

        print(f"  타겟 보유 현황: Blue {blue_with_targets}/{len(active_blue_troops)}, " + 
              f"Red {red_with_targets}/{len(active_red_troops)}")
        # ------------------

        # # for troop in self.troops:
        # #     if troop.alive:
        # #         if troop.team == "blue":
        # #             troop.assign_target(current_time, self.red_troops)
        # #         elif troop.team == "red":
        # #             troop.assign_target(current_time, self.blue_troops)

        # #!TEMP # active한 적만 필터링 >>>>
        # active_blue_troops = [t for t in self.blue_troops if getattr(t, 'active', False) and t.alive]
        # active_red_troops = [t for t in self.red_troops if getattr(t, 'active', False) and t.alive]
        # for troop in self.troops:
        #     if troop.alive and getattr(troop, 'active', False):
        #         if troop.team == "blue":
        #             troop.assign_target(current_time, active_red_troops)  # 변경!
        #         elif troop.team == "red":
        #             troop.assign_target(current_time, active_blue_troops)  # 변경!
        # #!TEMP # active한 적만 필터링 <<<<

    def assign_targets_for_nontarget_units(self, current_time):
        active_blue_troops = self.blue_observed
        active_red_troops = self.red_observed
        targets_assigned = 0

        for troop in self.troops:
            if troop.alive and getattr(troop, 'active', False) and troop.target is None:
                if troop.team == "blue":
                    troop.assign_target(current_time, active_red_troops)
                elif troop.team == "red":
                    troop.assign_target(current_time, active_blue_troops)

                # 새 타겟이 할당되었는지 확인
                if troop.target is not None:
                    targets_assigned += 1
                    print(f"  {troop.id}: 새 타겟 할당 -> {troop.target.id}")

        if targets_assigned > 0:
            print(f"  총 {targets_assigned}개 유닛에 새 타겟 할당됨")

    def get_enemy_list(self, troop):
        if troop.team == "blue":
            return self.red_troops
        elif troop.team == "red":
            return self.blue_troops
        else:
            print(f"[ERROR] wrong team affiliation: {troop.team}")
            return None

    def get_next_battle_time(self):
        next_battle_time = float("inf")
        for troop in self.troops:
            if troop.alive:
                if troop.next_fire_time < next_battle_time:
                    next_battle_time = troop.next_fire_time
        return next_battle_time

    def shuffle_troops(self):
        random.shuffle(self.troops)
        random.shuffle(self.blue_troops)
        random.shuffle(self.red_troops)

    def fire(self, current_time, history):

        self.shuffle_troops()

        # ========== 수정: 활성화된 부대만 사격 ==========
        firing_troops = [t for t in self.troops 
                        if (t.next_fire_time <= current_time and 
                            t.alive and 
                            getattr(t, 'active', False))]

        print(f"[{current_time:.1f}] {len(firing_troops)}개 유닛 사격 시도")

        for troop in firing_troops:
            # enemies = self.get_enemy_list(troop)
            enemies = self.get_observed_enemies(troop.team)
            troop.fire(current_time, enemies, self, history)

        # self.shuffle_troops()

        # for troop in self.troops:
        #     if troop.next_fire_time <= current_time:
        #         enemies = self.get_enemy_list(troop)
        #         troop.fire(current_time, enemies, history)

        # TODO: 에러 발생
        # if not self.main_ammo_int_changed and not self.secondary_ammo_int_changed:
        #     self.next_fire_time = round(current_time + 1.0, 2)
        #     return

    # def resupply(self, current_time):
    #     supply_trucks = [t for t in self.troops
    #                     if t.alive and t.type == UnitType.SUPPLY]

    #     for supply in supply_trucks:
    #         for unit in self.troops:
    #             if not unit.alive or unit.team != supply.team or unit.type == UnitType.SUPPLY:
    #                 continue
    #             if supply.get_distance(unit) < 0.3:
    #                 unit_name = unit.name
    #                 if unit_name not in SUPPLY_DATABASE or unit_name not in supply.supply_stock:
    #                     continue

    #                 max_ammo = AMMUNITION_DATABASE[unit_name].main_ammo + AMMUNITION_DATABASE[unit_name].secondary_ammo
    #                 curr_ammo = unit.main_ammo + unit.secondary_ammo
    #                 missing_ammo = max(0.0, max_ammo - curr_ammo)

    #                 available = supply.supply_stock[unit_name]
    #                 given = min(missing_ammo, available)

    #                 if given <= 0:
    #                     continue

    #                 # 보급 수행
    #                 unit_total = unit.main_ammo + unit.secondary_ammo
    #                 ratio_main = unit.main_ammo / unit_total if unit_total > 0 else 1.0

    #                 unit.main_ammo += given * ratio_main
    #                 unit.secondary_ammo += given * (1 - ratio_main)

    #                 # 보급 트럭 잔여량 감소
    #                 supply.supply_stock[unit_name] -= given

    #                 # 사격 제한 시간 갱신
    #                 unit.ammo_restricted_until = current_time + 5.0

    # ========== 추가 메서드: 전투 상황 분석 ==========
    def get_combat_status(self):
        """현재 전투 상황을 분석하여 반환"""
        active_blue = [t for t in self.blue_troops if getattr(t, 'active', False) and t.alive]
        active_red = [t for t in self.red_troops if getattr(t, 'active', False) and t.alive]

        blue_ready = len([t for t in active_blue if t.target and t.next_fire_time != float("inf")])
        red_ready = len([t for t in active_red if t.target and t.next_fire_time != float("inf")])

        return {
            'blue_active': len(active_blue),
            'red_active': len(active_red),
            'blue_ready_to_fire': blue_ready,
            'red_ready_to_fire': red_ready,
            'blue_tanks': len([t for t in active_blue if t.type == UnitType.TANK]),
            'red_tanks': len([t for t in active_red if t.type == UnitType.TANK])
        }


def terminate(troop_list:TroopList, current_time):
    # Check if all troops are dead or if the time limit is reached
    if current_time >= MAX_TIME:
        return True

    if not any(t.alive for t in troop_list.blue_troops) or not any(t.alive for t in troop_list.red_troops):
        print(f"[{current_time:.1f}] 전투 종료: 모든 부대가 파괴됨")
        return True

    for troop in troop_list.troops:
        if troop.type == UnitType.TANK and troop.alive:
            return False
        if troop.type == UnitType.APC and troop.alive:
            return False
    print(f"[{current_time:.1f}] 전투 종료: 모든 전차와 APC가 파괴됨")
    return True

#!TEMP >>>>
def update_troop_location_improved(troop_list: TroopList, battle_map, current_time):
    """개선된 부대 이동 업데이트"""

    for troop in troop_list.troops:
        if not troop.alive:
            continue
        
        # 활성화되지 않은 부대는 이동하지 않음
        if not troop.active or not troop.can_move:
            troop.update_velocity(Velocity(0, 0, 0))
            continue

        # 목적지 우선순위 결정
        if troop.fixed_dest:
            dest = troop.fixed_dest

            # 목적지 도달 확인
            dist_to_dest = math.hypot(
                troop.coord.x - dest.x,
                troop.coord.y - dest.y
            )
            
            # 목적지에 충분히 가까우면 정지
            if dist_to_dest < 5:  # 5픽셀(50m) 이내
                troop.update_velocity(Velocity(0, 0, 0))
                troop.fixed_dest = None  # 🔑 중요: 목적지 제거
                troop.can_move = False   # 🔑 중요: 이동 금지

                print(f"[{current_time:.1f}] {troop.id} 목적지 도달 완료")
                continue

        elif troop.target:
            # 전술적 목적지 계산 (측면공격, 매복 등)
            dest_coord = TacticalManager.get_tactical_destination(
                troop, troop.target, battle_map, troop_list.troops
            )
            dest = dest_coord
            # troop.can_move = True  # 🔑 중요: 활성화된 부대는 이동 가능
        else:
            dest = troop.coord  # 제자리

        # 개선된 이동 계산
        velocity = troop.compute_velocity_advanced(dest, battle_map, current_time)
        troop.update_velocity(velocity)
        troop.update_coord()

        # z 값을 DEM에서 업데이트
        xi, yi = int(troop.coord.x), int(troop.coord.y)
        if 0 <= yi < battle_map.height and 0 <= xi < battle_map.width:
            troop.coord.z = battle_map.dem_arr[yi, xi]

        # 맵 경계 벗어나면 제거
        if not (0 <= troop.coord.x < battle_map.width and 
                0 <= troop.coord.y < battle_map.height):
            troop.alive = False
#!TEMP <<<<
