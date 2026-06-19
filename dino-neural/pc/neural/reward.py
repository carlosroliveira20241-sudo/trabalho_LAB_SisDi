"""Cálculo de recompensa (função delegável `calcula_recompensa`, padrão PC).

Mantém a mesma fórmula do motor do jogo (physics.py) para coerência:
  +1 por frame sobrevivido, +10 ao passar obstáculo, -100 ao morrer.
Para NEAT, o fitness é o score total do episódio.
"""
from __future__ import annotations

from game import physics


def step_reward(passed_obstacle: bool, died: bool) -> float:
    r = physics.R_PER_FRAME
    if passed_obstacle:
        r += physics.R_PASS_OBSTACLE
    if died:
        r += physics.R_DEATH
    return r


def episode_fitness(score: int, frames: int) -> float:
    # TODO: combinar score e sobrevivência se quiser moldar melhor o NEAT
    return float(score)
