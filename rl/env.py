# env.py
# PettingZoo Parallel API 호환 MARL 환경 (외부 의존성 없음, DESIGN.md §3~6).
# 기존 시뮬 동역학(이동 물리/LOS/사격 Ph·Pk/관측)을 재사용하고, 결정(이동/타깃/교전)만 행동으로 받는다.

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import math
import contextlib
import numpy as np

from modules.troop import TroopList
from modules.map import Velocity

from rl.spaces import Box, MultiDiscrete
from rl.spawn import load_full_map, spawn_episode
from rl import obs as obsmod

# 이동 행동 9개: 0=정지, 1~8 = 8방향
_MOVES = [
    (0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
]

# 사격 결과 → 보상 크기 (DESIGN §6)
_RESULT_REWARD = {
    "catastrophic-kill": 1.0,
    "mobility-kill": 0.3,
    "firepower-kill": 0.3,
    "miss": 0.0,
}


@contextlib.contextmanager
def _silence():
    """기존 시뮬 코드의 디버그 print 억제 (RL 루프에서 stdout 폭주 방지)."""
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


class _EventLog:
    """troop.fire(...) 가 호출하는 history 인터페이스 스텁. 사격 이벤트만 수집."""
    def __init__(self):
        self.events = []  # (shooter_id, target_id, result_str)

    def add_to_battle_log(self, type_, shooter, target, target_type, result):
        self.events.append((shooter, target, result))


class WargameParallelEnv:
    """각 troop = 1 에이전트. 동시 행동(이산시간) Parallel 환경."""

    metadata = {"name": "wargame_marl_v0"}

    def __init__(self, map_size=120, decision_interval=5, max_decisions=60,
                 comp=None, seed=None):
        self.map_size = map_size
        self.decision_interval = decision_interval   # 정책 1결정 = 시뮬 N분
        self.max_decisions = max_decisions
        self.comp = comp
        self.full_map = load_full_map()
        self.rng = np.random.default_rng(seed)

        self._obs_space = Box(-1.0, 1.0, (obsmod.OBS_DIM,))
        self._act_space = MultiDiscrete([len(_MOVES), obsmod.ENEMY_K + 1, 2])

        self.possible_agents = []
        self.agents = []

    # --- PettingZoo 호환 space 접근자 ---
    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    # --- reset ---
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        from modules.troop import Troop
        Troop.counter = {}  # 에피소드마다 id 재현성 (B_TA1, R_TA1, ...)

        with _silence():
            self.cmap, troops = spawn_episode(
                self.full_map, self.rng, size=self.map_size, comp=self.comp)
            self.troop_list = TroopList(troops)
        # TroopList.__init__ 가 빈 관측목록으로 assign_targets 를 호출해 next_fire_time=inf 로
        # 만들어 두므로, 정책이 타깃을 정하기 전 초기화한다.
        for tr in self.troop_list.troops:
            tr.target = None
            tr.next_fire_time = 0.0

        self.t = 0.0
        self.n_decisions = 0
        self.evlog = _EventLog()
        self._ev_cursor = 0

        self.troops_by_id = {tr.id: tr for tr in self.troop_list.troops}
        self.agents = [tr.id for tr in self.troop_list.troops]
        self.possible_agents = list(self.agents)

        # 팀 초기 병력(종료 시 팀 보상 정규화용)
        self._init_count = {
            "blue": len(self.troop_list.blue_troops),
            "red": len(self.troop_list.red_troops),
        }

        with _silence():
            self.troop_list.update_observation(self.cmap)

        self._enemy_order = {}
        self._goal_dist = {}
        obs = {}
        for tr in self.troop_list.troops:
            v, order = obsmod.build_observation(tr, self.troop_list, self.cmap)
            obs[tr.id] = v
            self._enemy_order[tr.id] = order
            self._goal_dist[tr.id] = self._dist_to_goal(tr)
        infos = {a: {} for a in self.agents}
        return obs, infos

    # --- helpers ---
    def _dist_to_goal(self, tr):
        g = getattr(tr, "fixed_dest", None)   # 기존 시나리오 goal (RED만 보유)
        if g is None:
            return None
        return math.hypot(g.x - tr.coord.x, g.y - tr.coord.y)

    def _apply_move(self, tr, move_idx):
        dx, dy = _MOVES[move_idx]
        if dx == 0 and dy == 0:
            tr.update_velocity(Velocity(0, 0, 0))
            return
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        move_px = tr.calculate_movement_distance(self.cmap, self.t)  # 지형/주야 반영
        nx = tr.coord.x + ux * move_px
        ny = tr.coord.y + uy * move_px
        xi, yi = int(round(nx)), int(round(ny))
        if 0 <= xi < self.cmap.width and 0 <= yi < self.cmap.height and self.cmap.is_passable(xi, yi):
            tr.coord.x, tr.coord.y = nx, ny
            tr.coord.z = float(self.cmap.dem_arr[yi, xi])

    # --- step ---
    def step(self, actions):
        acting = list(self.agents)  # 이번 step 에 행동한 에이전트

        # 1) 정책 결정 적용: 타깃/교전/이동 의도 설정
        for aid in acting:
            tr = self.troops_by_id[aid]
            move_i, tgt_i, eng_i = (int(x) for x in actions[aid])
            order = self._enemy_order.get(aid, [])
            new_target = order[tgt_i] if (tgt_i < len(order) and order[tgt_i].alive) else None
            if new_target is not tr.target:
                tr.target = new_target
                if new_target is not None:
                    # 새 타깃 획득: 획득시간 t_a + 발사시간 t_f 뒤부터 사격 (assign_target 과 동일 공식)
                    tr.next_fire_time = round(self.t + tr.get_t_a() + tr.get_t_f(), 2)
                else:
                    tr.next_fire_time = float("inf")
            tr._engage = bool(eng_i)
            tr._move = move_i

        # 2) decision_interval 만큼 시뮬 진행 (이동 → 관측 → 사격)
        self._ev_cursor = len(self.evlog.events)
        with _silence():
            for _ in range(self.decision_interval):
                self.t = round(self.t + 1.0, 2)
                for tr in self.troop_list.troops:
                    if tr.alive and getattr(tr, "active", False):
                        self._apply_move(tr, getattr(tr, "_move", 0))
                self.troop_list.update_observation(self.cmap)
                for tr in self.troop_list.troops:
                    if not tr.alive or not getattr(tr, "_engage", False):
                        continue
                    if tr.target is None or not tr.target.alive:
                        continue
                    if tr.get_distance(tr.target) > tr.range_km:
                        continue
                    if tr.next_fire_time <= self.t:
                        # enemy_list=[] → 명중/살상 후 규칙기반 재타게팅 억제(정책이 다음 결정에서 재설정)
                        tr.fire(self.t, [], self.troop_list, self.evlog)
                self.troop_list.remove_dead_troops()

        # 3) 보상 계산
        rewards = {a: 0.0 for a in acting}
        new_events = self.evlog.events[self._ev_cursor:]
        for shooter, target, result in new_events:
            r = _RESULT_REWARD.get(result, 0.0)
            if r <= 0:
                continue
            if shooter in rewards:
                rewards[shooter] += r           # 가한 피해
            if target in rewards:
                rewards[target] -= r            # 받은 피해
        # 사망 페널티 + goal 접근 보상
        alive_ids = {tr.id for tr in self.troop_list.troops if tr.alive}
        for aid in acting:
            tr = self.troops_by_id[aid]
            if aid not in alive_ids:
                rewards[aid] -= 1.0             # 사망
            else:
                prev = self._goal_dist.get(aid)
                cur = self._dist_to_goal(tr)
                if prev is not None and cur is not None:   # goal 있는 RED만 진척 보상
                    rewards[aid] += 0.01 * (prev - cur)
                    self._goal_dist[aid] = cur

        # 4) 종료/절단 판정
        self.n_decisions += 1
        blue_alive = any(tr.alive for tr in self.troop_list.blue_troops)
        red_alive = any(tr.alive for tr in self.troop_list.red_troops)
        team_done = (not blue_alive) or (not red_alive)
        time_up = self.n_decisions >= self.max_decisions
        episode_over = team_done or time_up

        # 종료 시 팀 보상 (승: 잔존비례, 패: -)
        if episode_over:
            for aid in acting:
                tr = self.troops_by_id[aid]
                won = (tr.team == "blue" and not red_alive) or (tr.team == "red" and not blue_alive)
                lost = (tr.team == "blue" and not blue_alive) or (tr.team == "red" and not red_alive)
                if won:
                    rewards[aid] += 1.0
                elif lost:
                    rewards[aid] -= 1.0

        terminations = {a: False for a in acting}
        truncations = {a: False for a in acting}
        for aid in acting:
            dead = aid not in alive_ids
            if dead or team_done:
                terminations[aid] = True
            if time_up and not dead and not team_done:
                truncations[aid] = True

        # 5) 다음 관측 (살아있고 에피소드 진행 중인 에이전트만)
        obs = {}
        if not episode_over:
            with _silence():
                self.troop_list.update_observation(self.cmap)
            next_agents = []
            for tr in self.troop_list.troops:
                if not tr.alive:
                    continue
                v, order = obsmod.build_observation(tr, self.troop_list, self.cmap)
                obs[tr.id] = v
                self._enemy_order[tr.id] = order
                next_agents.append(tr.id)
            self.agents = next_agents
        else:
            self.agents = []

        infos = {a: {} for a in acting}
        return obs, rewards, terminations, truncations, infos
