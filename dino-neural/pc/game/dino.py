"""O dinossauro: estado e física de pulo."""
from __future__ import annotations

from . import physics


class Dino:
    def __init__(self):
        self.x = physics.DINO_X
        self.y = float(physics.GROUND_Y - physics.DINO_H)
        self.vy = 0.0
        self.on_ground = True
        self.ducking = False

    def jump(self) -> None:
        if self.on_ground:
            self.vy = physics.JUMP_IMPULSE
            self.on_ground = False

    def duck(self, active: bool) -> None:
        """Agacha: reduz a hitbox pela metade (alinhada ao chão). No ar, a
        queda fica mais rápida (fast-fall) — tratado em update()."""
        self.ducking = active

    def update(self) -> None:
        self.vy += physics.GRAVITY
        if self.ducking and not self.on_ground:
            self.vy += physics.DUCK_GRAVITY_EXTRA      # cai mais rápido agachado
        self.y += self.vy
        floor = physics.GROUND_Y - physics.DINO_H
        if self.y >= floor:
            self.y = float(floor)
            self.vy = 0.0
            self.on_ground = True

    @property
    def hitbox(self):
        if self.ducking:
            h = physics.DINO_DUCK_H
            bottom = self.y + physics.DINO_H           # mantém o pé no mesmo lugar
            return (self.x, bottom - h, physics.DINO_W, h)
        return (self.x, self.y, physics.DINO_W, physics.DINO_H)
