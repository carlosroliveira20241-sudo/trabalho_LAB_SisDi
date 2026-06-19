"""DQN — Deep Q-Network. Roda no PC (replay buffer não cabe nos 2 KB do Uno).

Quando delegado, só o `forward_pass` pode ir ao Arduino; treino (replay +
target net) fica sempre no PC.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np

GAMMA = 0.99
BATCH = 32
BUFFER_SIZE = 5000


class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch: int = BATCH):
        return random.sample(self.buf, min(batch, len(self.buf)))

    def __len__(self):
        return len(self.buf)


class DQN:
    def __init__(self, router, net, target_net, lr: float = 0.001):
        self.router = router
        self.net = net
        self.target = target_net
        self.lr = lr
        self.buffer = ReplayBuffer()
        self.epsilon = 1.0

    def act(self, state_vec) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        q = self.router.call("forward_pass", state_vec)  # pode ir ao Arduino
        return int(np.argmax(q))

    def train_step(self) -> None:
        # TODO: amostrar batch, calcular alvo r + gamma*max Q_target, MSE, passo
        ...
