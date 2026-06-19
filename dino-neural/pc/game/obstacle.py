"""Obstáculos: cactos (pequeno, grande, duplo) e pterodáctilo (aéreo)."""
from __future__ import annotations

import random
from enum import Enum

from . import physics


class Kind(Enum):
    CACTUS_SMALL = 0
    CACTUS_LARGE = 1
    CACTUS_DOUBLE = 2
    PTERODACTYL = 3        # voa alto: desvia ficando no chão (não pular)
    PTERODACTYL_LOW = 4    # voa na altura da cabeça: desvia AGACHANDO


# (largura, altura, offset_y a partir do chão) por tipo
DIMENSIONS = {
    Kind.CACTUS_SMALL:   (20, 40, 0),
    Kind.CACTUS_LARGE:   (28, 55, 0),
    Kind.CACTUS_DOUBLE:  (48, 55, 0),
    Kind.PTERODACTYL:    (42, 32, 60),   # y=158..190: passa por baixo de pé
    Kind.PTERODACTYL_LOW: (42, 100, 30),  # y=120..220: alto o bastante p/ bater
                                          # de pé E no ápice do pulo; só passa o
                                          # dino agachado (top=226 > 220)
}


class Obstacle:
    def __init__(self, kind: Kind, x: float):
        self.kind = kind
        self.x = x
        w, h, off = DIMENSIONS[kind]
        self.w, self.h = w, h
        self.y = physics.GROUND_Y - h - off
        self.passed = False  # já contabilizou recompensa de "passou"

    def update(self, speed: float) -> None:
        self.x -= speed

    @property
    def offscreen(self) -> bool:
        return self.x + self.w < 0

    @property
    def hitbox(self):
        return (self.x, self.y, self.w, self.h)

    @staticmethod
    def random(x: float, kinds: list["Kind"] | None = None) -> "Obstacle":
        """Cria um obstáculo de tipo aleatório. `kinds` restringe os tipos
        possíveis (ex.: só cactos no currículo inicial)."""
        return Obstacle(random.choice(kinds or list(Kind)), x)


# atalhos de currículo
CACTUS_ONLY = [Kind.CACTUS_SMALL, Kind.CACTUS_LARGE, Kind.CACTUS_DOUBLE]
# cactos (pular) + obstáculo baixo (agachar): treina o agachar sem o ptero alto
CACTUS_AND_DUCK = CACTUS_ONLY + [Kind.PTERODACTYL_LOW]
ALL_KINDS = list(Kind)
