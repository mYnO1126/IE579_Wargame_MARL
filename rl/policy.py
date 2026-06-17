# policy.py
# 파라미터 공유 Actor-Critic (MLP). MultiDiscrete 행동 [move, target, engage].
# 관측이 8방향 압축 지형 + top-k 적/아군이라 MLP로 충분(CNN 불필요).

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_nvec, hidden=128):
        super().__init__()
        self.act_nvec = list(int(n) for n in act_nvec)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, sum(self.act_nvec))   # 3개 헤드 logits 연결
        self.critic = nn.Linear(hidden, 1)

    def _dists(self, h):
        logits = self.actor(h)
        dists, i = [], 0
        for n in self.act_nvec:
            dists.append(Categorical(logits=logits[..., i:i + n]))
            i += n
        return dists

    @torch.no_grad()
    def act(self, obs):
        """행동 샘플링. obs: (B, obs_dim) → actions (B, 3), logp (B,), value (B,)."""
        h = self.trunk(obs)
        dists = self._dists(h)
        acts = torch.stack([d.sample() for d in dists], dim=-1)
        logp = torch.stack([d.log_prob(acts[..., k]) for k, d in enumerate(dists)], dim=-1).sum(-1)
        val = self.critic(h).squeeze(-1)
        return acts, logp, val

    def evaluate(self, obs, acts):
        """PPO 업데이트용: 주어진 행동의 logp, entropy, value."""
        h = self.trunk(obs)
        dists = self._dists(h)
        logp = torch.stack([d.log_prob(acts[..., k]) for k, d in enumerate(dists)], dim=-1).sum(-1)
        ent = torch.stack([d.entropy() for d in dists], dim=-1).sum(-1)
        val = self.critic(h).squeeze(-1)
        return logp, ent, val
