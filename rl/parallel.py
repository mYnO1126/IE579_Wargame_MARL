# parallel.py
# 멀티프로세스 롤아웃 수집기. worker N개가 각자 env+정책 사본으로 롤아웃을 모으고,
# 메인이 합쳐 PPO 업데이트한다. (CPU 바운드 env라 진짜 병렬화는 멀티프로세스가 필요)

import os
import io
import multiprocessing as mp

import numpy as np
import torch

# worker 프로세스의 전역 상태(프로세스당 env/policy 1개, 1회 생성)
_W = {}


def _init_worker(team, obs_dim, act_nvec, global_dim, base_seed, idx_counter):
    import torch
    torch.set_num_threads(1)   # 프로세스마다 1스레드 → 오버서브스크립션 방지
    from rl.env import WargameParallelEnv
    from rl.policy import ActorCritic
    # 프로세스마다 다른 seed (다양한 에피소드)
    wid = idx_counter.value
    with idx_counter.get_lock():
        idx_counter.value += 1
    _W["env"] = WargameParallelEnv(seed=base_seed + 1000 * (wid + 1))
    _W["policy"] = ActorCritic(obs_dim, act_nvec, global_dim)
    _W["policy"].eval()
    _W["team"] = team


def _collect_task(args):
    state_bytes, steps = args
    from rl.rollout import collect
    sd = torch.load(io.BytesIO(state_bytes), map_location="cpu")
    _W["policy"].load_state_dict(sd)
    return collect(_W["env"], _W["policy"], _W["team"], steps, device="cpu")


class ParallelCollector:
    """worker N개로 롤아웃을 병렬 수집."""

    def __init__(self, n_workers, team, obs_dim, act_nvec, global_dim, seed=0):
        self.n = n_workers
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        self.pool = ctx.Pool(
            n_workers, initializer=_init_worker,
            initargs=(team, obs_dim, act_nvec, global_dim, seed, counter),
        )

    def collect(self, policy, roll_steps):
        buf = io.BytesIO()
        torch.save(policy.state_dict(), buf)
        sb = buf.getvalue()
        per = max(1, roll_steps // self.n)
        results = self.pool.map(_collect_task, [(sb, per)] * self.n)

        B = {"obs": [], "glob": [], "act": [], "logp": [], "adv": [], "ret": []}
        ret_sum, win_sum, games, steps = 0.0, 0.0, 0, 0
        for b, st in results:
            for k in B:
                B[k].extend(b[k])
            ret_sum += st["ep_return"] * st["games"]
            win_sum += st["win_rate"] * st["games"]
            games += st["games"]
            steps += st["agent_steps"]
        stats = {"ep_return": ret_sum / max(1, games), "win_rate": win_sum / max(1, games),
                 "games": games, "agent_steps": steps}
        return B, stats

    def close(self):
        self.pool.close()
        self.pool.join()
