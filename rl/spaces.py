# spaces.py
# gymnasium 의존성 없이 쓰는 최소 space 정의 (PettingZoo Parallel API 호환용).
# 나중에 gymnasium 을 쓰게 되면 이걸 gymnasium.spaces 로 교체하면 됨.

import numpy as np


class Box:
    """연속 관측 공간."""
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = tuple(shape)
        self.dtype = dtype

    def sample(self, rng):
        return rng.uniform(self.low, self.high, size=self.shape).astype(self.dtype)


class MultiDiscrete:
    """여러 개의 이산 행동 헤드. nvec = 각 헤드의 선택지 개수."""
    def __init__(self, nvec):
        self.nvec = list(int(n) for n in nvec)

    def sample(self, rng):
        return np.array([rng.integers(0, n) for n in self.nvec], dtype=np.int64)
