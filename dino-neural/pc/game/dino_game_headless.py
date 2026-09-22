"""Motor do jogo SEM render — a fonte de verdade da simulação.

Esta classe contém toda a lógica (física, spawn de obstáculos, colisão, score)
e expõe o estado interno direto, sem overhead. É usada:
  - diretamente pelo NEAT (muitas instâncias simultâneas, sem gráficos);
  - como núcleo do `dino_game.py`, que só adiciona render por cima.

Estado exposto (game_state):
  distance_to_obstacle, obstacle_height, current_speed, dino_y, is_dead, score
"""
from __future__ import annotations

from . import physics
from .dino import Dino
from .obstacle import Obstacle

# ações
ACTION_NONE = 0
ACTION_JUMP = 1
ACTION_DUCK = 2

SPAWN_GAP = 350  # distância mínima entre obstáculos (px)


class DinoGameHeadless:
    def __init__(self, seed: int | None = None, kinds=None):
        # `kinds` restringe os tipos de obstáculo (currículo); None = todos.
        # TODO: usar seed para reprodutibilidade entre instâncias NEAT
        self.kinds = kinds
        self.reset()

    def reset(self) -> None:
        self.dino = Dino()
        self.obstacles: list[Obstacle] = [Obstacle.random(physics.WIDTH, self.kinds)]
        self.speed = physics.SPEED_START
        self.score = 0
        self.frames = 0
        self.is_dead = False
        self.last_reward = 0.0

    # --- um passo da simulação ----------------------------------------------
    def step(self, action: int) -> float:
        """Aplica a ação, avança um frame e devolve a recompensa do frame."""
        if self.is_dead:
            return 0.0

        reward = physics.R_PER_FRAME
        self.dino.duck(action == ACTION_DUCK)   # agacha/desagacha a cada frame
        if action == ACTION_JUMP:
            if self.dino.on_ground:             # só conta pulo que de fato acontece
                reward += physics.R_JUMP
            self.dino.jump()

        self.dino.update()

        # velocidade crescente
        self.speed = min(physics.SPEED_MAX, self.speed + physics.SPEED_ACCEL)

        for obs in self.obstacles:
            obs.update(self.speed)
            if not obs.passed and obs.x + obs.w < self.dino.x:
                obs.passed = True
                self.score += 1
                reward += physics.R_PASS_OBSTACLE

        self._cull_and_spawn()

        if self._collided():
            self.is_dead = True
            reward += physics.R_DEATH

        self.frames += 1
        self.last_reward = reward
        return reward

    def _cull_and_spawn(self) -> None:
        self.obstacles = [o for o in self.obstacles if not o.offscreen]
        last = max(self.obstacles, key=lambda o: o.x, default=None)
        if last is None or physics.WIDTH - last.x >= SPAWN_GAP:
            self.obstacles.append(Obstacle.random(physics.WIDTH + 50, self.kinds))

    def _collided(self) -> bool:
        for o in self.obstacles:
            if physics.aabb_collision(*self.dino.hitbox, *o.hitbox):
                return True
        return False

    # --- estado para a rede neural ------------------------------------------
    def _next_obstacle(self) -> Obstacle | None:
        ahead = [o for o in self.obstacles if o.x + o.w >= self.dino.x]
        return min(ahead, key=lambda o: o.x, default=None)

    def game_state(self) -> dict:
        nxt = self._next_obstacle()
        return {
            "distance_to_obstacle": (nxt.x - self.dino.x) if nxt else float(physics.WIDTH),
            "obstacle_height": float(nxt.h) if nxt else 0.0,
            "current_speed": self.speed,
            "dino_y": self.dino.y,
            "is_dead": self.is_dead,
            "score": self.score,
        }

    def features(self):
        """As 4 entradas da rede 4->4->3 (e do NEAT): distância, velocidade e a
        posição vertical (topo e base) do próximo obstáculo — o que permite
        distinguir cacto (chão, pular) de voador (aéreo, agachar)."""
        o = self._next_obstacle()
        if o is None:
            return [1.0, self.speed / physics.SPEED_MAX, 1.0, 1.0]
        return [
            (o.x - self.dino.x) / physics.WIDTH,
            self.speed / physics.SPEED_MAX,
            o.y / physics.HEIGHT,
            (o.y + o.h) / physics.HEIGHT,
        ]
