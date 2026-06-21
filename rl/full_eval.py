# full_eval.py
# 풀 시뮬 통합 비교 (DESIGN §10). 전체 시나리오(PLACEMENT/TIMELINE/풀맵)를 돌리면서
# 한 팀의 결정(이동/타깃/교전)을 [규칙기반] 또는 [학습 정책]으로 바꿔 끼워 결과를 비교한다.
# 물리/사격(Ph·Pk)/LOS/관측/TIMELINE/상대팀은 main.py 와 동일하게 유지.
#
# ※ caveat: 현재 정책은 작은 크롭(4v4)에서 학습 + obs 거리정규화가 맵크기 종속(dmax) →
#   풀맵(322유닛)으로의 전이 격차가 큼. 결과 해석 시 유의(NOTES[G], DESIGN §10).
#
# 실행:  conda run -n wargame python -m rl.full_eval --ckpt rl/policies/ippo_blue.pt --seeds 2
#        (기본 max_time=2880분 = 원래 시뮬 끝까지/자연종료. --max_time 으로 줄이면 스모크)

import os
import sys
import math
import random
import argparse
import contextlib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

import modules.troop as troop_mod
from modules.map import Map, Velocity
from modules.placement import PLACEMENT, grid_sample_no_overlap
from modules.timeline import TIMELINE
from modules.troop import TroopList, update_troop_location_improved, terminate
from main import create_from_positions, handle_event

from rl.env import _MOVES
from rl.policy import ActorCritic
from rl import obs as obsmod

_NPZ = os.path.join(_ROOT, "map", "golan_full_dataset_cropped.npz")


@contextlib.contextmanager
def _silence():
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


class _StubHist:
    """fire()가 부르는 history 인터페이스 스텁(로그 불필요)."""
    def add_to_battle_log(self, *a, **k):
        pass


def _setup(seed, battle_map):
    """main.py 와 동일하게 PLACEMENT 샘플링 + troops 생성."""
    random.seed(seed)
    np.random.seed(seed)
    used = set()
    for team, affs in PLACEMENT.items():
        for affiliation, feat in affs.items():
            x_range, y_range = feat["loc"]
            feat["locs"] = []
            has_goal = "dest" in feat
            if has_goal:
                gx_range, gy_range = feat["dest"]
                feat["goals"] = []
            for comp, cnt in feat["comp"].items():
                min_gap = 4 if comp == "AK-47" else 6
                coords = grid_sample_no_overlap(x_range, y_range, cnt, min_gap=min_gap, used=used)
                feat["locs"].extend([(x, y, battle_map.dem_arr[y, x]) for x, y, _ in coords])
            if has_goal:
                goals = grid_sample_no_overlap(gx_range, gy_range, 10, min_gap=6, used=set())
                feat["goals"].extend([(x, y, float(battle_map.dem_arr[y, x])) for x, y, _ in goals])
    return TroopList(troop_list=create_from_positions(PLACEMENT))


def _policy_move(tr, battle_map, t):
    dx, dy = _MOVES[getattr(tr, "_move", 0)]
    if dx == 0 and dy == 0:
        tr.update_velocity(Velocity(0, 0, 0)); return
    norm = math.hypot(dx, dy); ux, uy = dx / norm, dy / norm
    move_px = tr.calculate_movement_distance(battle_map, t)
    nx, ny = tr.coord.x + ux * move_px, tr.coord.y + uy * move_px
    xi, yi = int(round(nx)), int(round(ny))
    if 0 <= xi < battle_map.width and 0 <= yi < battle_map.height and battle_map.is_passable(xi, yi):
        tr.coord.x, tr.coord.y, tr.coord.z = nx, ny, float(battle_map.dem_arr[yi, xi])
        tr.update_velocity(Velocity(ux * move_px, uy * move_px, 0))
    else:
        tr.update_velocity(Velocity(0, 0, 0))


def _decide(troop_list, battle_map, tm, policy, t, device):
    """정책팀 tm 의 활성 유닛에 대해 정책으로 이동/타깃/교전 결정."""
    learn = [tr for tr in troop_list.troops
             if tr.alive and tr.team == tm and getattr(tr, "active", False)]
    if not learn:
        return
    vecs, orders = [], []
    for tr in learn:
        v, order = obsmod.build_observation(tr, troop_list, battle_map)
        vecs.append(v); orders.append(order)
    acts, _ = policy.act(torch.as_tensor(np.stack(vecs), dtype=torch.float32, device=device),
                         deterministic=True)
    acts = acts.cpu().numpy()
    for i, tr in enumerate(learn):
        mv, ti, eng = int(acts[i][0]), int(acts[i][1]), int(acts[i][2])
        order = orders[i]
        new_t = order[ti] if (ti < len(order) and order[ti].alive) else None
        if new_t is not tr.target:
            tr.target = new_t
            tr.next_fire_time = round(t + tr.get_t_a() + tr.get_t_f(), 2) if new_t else float("inf")
        tr._move, tr._engage = mv, bool(eng)


def _team_fire(troop_list, policy_teams, t, hist):
    """정책팀=engage+유효타깃일 때만(규칙 재타게팅 억제), 규칙팀=평소대로."""
    troop_list.shuffle_troops()
    due = [tr for tr in troop_list.troops if tr.next_fire_time <= t and tr.alive and getattr(tr, "active", False)]
    for tr in due:
        if tr.team in policy_teams:
            if (getattr(tr, "_engage", False) and tr.target and tr.target.alive
                    and tr.get_distance(tr.target) <= tr.range_km):
                tr.fire(t, [], troop_list, hist)
        else:
            tr.fire(t, troop_list.get_observed_enemies(tr.team), troop_list, hist)


def run_full(seed, policies, max_time=2880.0, decision_interval=5, device="cpu"):
    """policies = {"blue": policy|None, "red": policy|None}. 정책 있는 팀은 정책, 없으면 규칙기반.
    policies가 모두 None 이면 all-rule 베이스라인."""
    policy_teams = [tm for tm in ("blue", "red") if policies.get(tm) is not None]
    troop_mod.MAX_TIME = max_time
    with _silence():                       # TroopList.__init__의 assign_targets print 억제
        battle_map = Map(filename=_NPZ)
        troop_list = _setup(seed, battle_map)
    init_b, init_r = len(troop_list.blue_troops), len(troop_list.red_troops)
    hist = _StubHist()
    t = 0.0
    timeline_index = 0

    with _silence():
        while True:
            if timeline_index < len(TIMELINE) and t == TIMELINE[timeline_index].time:
                handle_event(TIMELINE[timeline_index], troop_list, battle_map)
                timeline_index += 1
            troop_list.remove_dead_troops()
            if terminate(troop_list=troop_list, current_time=t):
                break
            t = round(t + 1.0, 2)

            # ---- 정책 결정(결정주기마다, 각 정책팀) ----
            if policy_teams and round(t) % decision_interval == 0:
                troop_list.update_observation(battle_map)
                for tm in policy_teams:
                    _decide(troop_list, battle_map, tm, policies[tm], t, device)

            # ---- 이동: 규칙팀=규칙 이동, 정책팀=정책 방향 ----
            if not policy_teams:
                update_troop_location_improved(troop_list, battle_map, t)
            else:
                saved = [(tr, tr.can_move) for tr in troop_list.troops if tr.team in policy_teams]
                for tr, _cm in saved:
                    tr.can_move = False                      # 규칙 이동에서 정책팀 제외
                update_troop_location_improved(troop_list, battle_map, t)   # 규칙팀만 이동
                for tr, cm in saved:
                    tr.can_move = cm
                for tr in troop_list.troops:
                    if tr.alive and tr.team in policy_teams and getattr(tr, "active", False) and tr.can_move:
                        _policy_move(tr, battle_map, t)

            # ---- 관측 → 타게팅(규칙은 무타깃 유닛만 채움) → 사격 ----
            troop_list.update_observation(battle_map)
            troop_list.assign_targets_for_nontarget_units(t)
            if not policy_teams:
                if troop_list.get_next_battle_time() <= t:
                    troop_list.fire(t, hist)
            else:
                _team_fire(troop_list, policy_teams, t, hist)

    b = sum(tr.alive for tr in troop_list.blue_troops)
    r = sum(tr.alive for tr in troop_list.red_troops)
    return {"blue": b, "red": r, "blue_init": init_b, "red_init": init_r, "time": t}


def _load(path):
    if not path:
        return None
    ck = torch.load(path, map_location="cpu")
    p = ActorCritic(ck["obs_dim"], ck["act_nvec"], ck["global_dim"])
    p.load_state_dict(ck["model"]); p.eval()
    return p


def _team_metrics(run, tm):
    if tm == "blue":
        of, oi, ef, ei = run["blue"], run["blue_init"], run["red"], run["red_init"]
    else:
        of, oi, ef, ei = run["red"], run["red_init"], run["blue"], run["blue_init"]
    ol, el = oi - of, ei - ef
    return {"LER": el / max(1, ol), "생존율": of / max(1, oi), "손실": ol, "적손실": el}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blue", default=None, help="blue 정책 ckpt (없으면 blue=규칙기반)")
    ap.add_argument("--red", default=None, help="red 정책 ckpt (없으면 red=규칙기반)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--max_time", type=float, default=2880.0,
                    help="기본 2880분 = 원래 시뮬 끝까지(또는 자연종료). 빠른 스모크용으로만 줄여 쓸 것")
    ap.add_argument("--out", default=None, help="결과(seed별 raw + 팀별 metric)를 JSON으로 저장할 경로")
    args = ap.parse_args()
    if not args.blue and not args.red:
        ap.error("최소 한 쪽(--blue/--red) 체크포인트가 필요합니다.")

    policies = {"blue": _load(args.blue), "red": _load(args.red)}
    pol_sides = [t for t in ("blue", "red") if policies[t] is not None]
    print(f"풀 시뮬 통합 비교: 정책팀={pol_sides}  seeds={args.seeds}  max_time={args.max_time:.0f}min")
    print("(비교 대상 = all-rule 베이스라인. 정책팀만 정책으로 교체)\n")

    base_runs, cfg_runs = [], []
    for s in range(args.seeds):
        base_runs.append(run_full(s, {}, args.max_time))           # all-rule 베이스라인
        cfg_runs.append(run_full(s, policies, args.max_time))      # 지정 정책 구성
        print(f"  seed {s}: base(blue={base_runs[-1]['blue']},red={base_runs[-1]['red']}) | "
              f"cfg(blue={cfg_runs[-1]['blue']},red={cfg_runs[-1]['red']})")

    summary = {}                                                    # JSON 저장용
    for tm in ("blue", "red"):
        tag = "정책" if tm in pol_sides else "규칙(고정)"
        bm = [_team_metrics(r, tm) for r in base_runs]
        cm = [_team_metrics(r, tm) for r in cfg_runs]
        print(f"\n[{tm} — {tag}]   {'metric':<8}{'rule(mean±std)':>17}{'config(mean±std)':>18}{'Δ paired':>14}")
        print("-" * 60)
        summary[tm] = {"tag": tag}
        for k in bm[0]:
            rv = np.array([m[k] for m in bm]); pv = np.array([m[k] for m in cm]); dv = pv - rv
            print(f"{'':<11}{k:<8}{rv.mean():>10.2f}±{rv.std():<6.2f}{pv.mean():>9.2f}±{pv.std():<6.2f}"
                  f"{dv.mean():>+8.2f}±{dv.std():.2f}")
            summary[tm][k] = {"rule_mean": float(rv.mean()), "rule_std": float(rv.std()),
                              "cfg_mean": float(pv.mean()), "cfg_std": float(pv.std()),
                              "delta_mean": float(dv.mean()), "delta_std": float(dv.std())}
    print("\n해석: 정책팀의 LER↑/생존율↑/적손실↑ 이면 규칙 대비 개선. (LER=적손실/아군손실=교환비)")

    if args.out:
        import json
        payload = {"policy_sides": pol_sides, "seeds": args.seeds, "max_time": args.max_time,
                   "base_runs": base_runs, "cfg_runs": cfg_runs, "summary": summary}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
