"""Constantes de física e colisão do jogo.

Centraliza os parâmetros para que o jogo com render e o headless (NEAT) usem
exatamente a mesma dinâmica.
"""
from __future__ import annotations

# tela
WIDTH, HEIGHT = 800, 300
GROUND_Y = 250

# dino
DINO_X = 80
DINO_W, DINO_H = 44, 47
DINO_DUCK_H = 24       # altura agachado (~metade) — alinhado pelo chão
GRAVITY = 0.9
JUMP_IMPULSE = -15.0   # velocidade vertical inicial do pulo (y cresce p/ baixo)
DUCK_GRAVITY_EXTRA = 1.8  # queda mais rápida quando agacha no ar (fast-fall)

# velocidade do jogo
SPEED_START = 6.0
SPEED_MAX = 18.0
SPEED_ACCEL = 0.001    # incremento por frame

# recompensa (espelha neural/reward.py)
R_PER_FRAME = 1.0
R_PASS_OBSTACLE = 10.0
R_DEATH = -100.0
R_JUMP = 0.0           # custo de pulo (0 = desligado; ver notas de treino)


def aabb_collision(ax, ay, aw, ah, bx, by, bw, bh) -> bool:
    """Colisão por hitbox (axis-aligned bounding box)."""
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by
