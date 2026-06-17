# policy.py
# MAPPO(CTDE): 분산 실행 actor + 중앙 critic.
#  - actor:  로컬 관측 → 행동 분포 (실행 시 로컬 정보만 사용)
#  - critic: 로컬 관측 ⊕ 전역 상태 → 가치 (학습 시 전역 정보로 팀 가치를 잘 추정 → 분산↓)
# 행동: MultiDiscrete [move, target, engage]. 파라미터 공유(한 팀 = 하나의 망).

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_nvec, global_dim, hidden=128):
        super().__init__()
        self.act_nvec = list(int(n) for n in act_nvec)
        self.global_dim = int(global_dim)

        # actor (로컬 관측만)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, sum(self.act_nvec)),
        )
        # 중앙 critic (로컬 관측 ⊕ 전역 상태)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim + self.global_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def _dists(self, logits):
        dists, i = [], 0
        for n in self.act_nvec:
            dists.append(Categorical(logits=logits[..., i:i + n]))
            i += n
        return dists

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """행동 선택(actor만). obs (B, obs_dim) → acts (B,3), logp (B,).
        deterministic=True 면 각 헤드 argmax(평가용), 아니면 샘플링(학습용)."""
        dists = self._dists(self.actor(obs))
        if deterministic:
            acts = torch.stack([torch.argmax(d.probs, dim=-1) for d in dists], dim=-1)
        else:
            acts = torch.stack([d.sample() for d in dists], dim=-1)
        logp = torch.stack([d.log_prob(acts[..., k]) for k, d in enumerate(dists)], dim=-1).sum(-1)
        return acts, logp

    @torch.no_grad()
    def get_value(self, obs, glob):
        """중앙 critic 가치. obs (B,obs_dim), glob (B,global_dim) → (B,)."""
        return self.critic(torch.cat([obs, glob], dim=-1)).squeeze(-1)

    def evaluate(self, obs, glob, acts):
        """PPO 업데이트용: logp, entropy, value."""
        dists = self._dists(self.actor(obs))
        logp = torch.stack([d.log_prob(acts[..., k]) for k, d in enumerate(dists)], dim=-1).sum(-1)
        ent = torch.stack([d.entropy() for d in dists], dim=-1).sum(-1)
        val = self.critic(torch.cat([obs, glob], dim=-1)).squeeze(-1)
        return logp, ent, val
