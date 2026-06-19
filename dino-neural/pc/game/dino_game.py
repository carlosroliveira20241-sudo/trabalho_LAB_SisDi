"""Jogo do dino COM render (Pygame), por cima do motor headless.

Roda em thread própria, no seu próprio FPS, desacoplado do serial/benchmark
(ver README). A IA escreve a próxima ação em `self.pending_action`; o loop a
consome no próximo frame. Modos: humano, IA, híbrido.
"""
from __future__ import annotations

import threading

import pygame

from . import physics
from .dino_game_headless import DinoGameHeadless, ACTION_NONE, ACTION_JUMP, ACTION_DUCK

MODE_HUMAN = "human"
MODE_AI = "ai"
MODE_HYBRID = "hybrid"


class DinoGame:
    def __init__(self, mode: str = MODE_AI, fps: int = 60):
        self.engine = DinoGameHeadless()
        self.mode = mode
        self.fps = fps
        self.pending_action = ACTION_NONE  # setado pela thread da IA
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # --- controle de thread --------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def set_action(self, action: int) -> None:
        with self._lock:
            self.pending_action = action

    def game_state(self) -> dict:
        return self.engine.game_state()

    # --- entrada pública -----------------------------------------------------
    def run_blocking(self) -> None:
        """Roda o loop na thread atual (use para jogar no modo humano)."""
        self._running = True
        self._loop()

    # --- loop do pygame ------------------------------------------------------
    def _loop(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode((physics.WIDTH, physics.HEIGHT))
        pygame.display.set_caption("Dino Neural")
        self._font = pygame.font.SysFont("consolas", 18)
        clock = pygame.time.Clock()

        while self._running:
            action = self._handle_input()
            self.engine.step(action)
            self._render(screen)
            if self.engine.is_dead:
                # TODO: notificar o trainer (fim de episódio) antes do reset
                self._render_gameover(screen)
                if self._wait_restart():
                    self.engine.reset()
            clock.tick(self.fps)

        pygame.quit()

    def _wait_restart(self) -> bool:
        """No modo humano, espera ESPACO para reiniciar; na IA, reinicia já."""
        if self.mode == MODE_AI:
            return True
        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return True
            pygame.time.wait(20)
        return False

    def _handle_input(self) -> int:
        action = ACTION_NONE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
        keys = pygame.key.get_pressed()
        if self.mode in (MODE_HUMAN, MODE_HYBRID):
            if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
                action = ACTION_JUMP
            elif keys[pygame.K_DOWN]:           # agachar
                action = ACTION_DUCK
        if self.mode in (MODE_AI, MODE_HYBRID):
            with self._lock:
                if self.pending_action != ACTION_NONE:
                    action = self.pending_action
                    self.pending_action = ACTION_NONE
        return action

    @staticmethod
    def _irect(hitbox):
        return [int(v) for v in hitbox]

    def _render(self, screen) -> None:
        from .obstacle import Kind
        screen.fill((247, 247, 247))
        pygame.draw.line(screen, (83, 83, 83), (0, physics.GROUND_Y),
                         (physics.WIDTH, physics.GROUND_Y), 2)
        d = self.engine.dino
        pygame.draw.rect(screen, (83, 83, 83), self._irect(d.hitbox))
        for o in self.engine.obstacles:
            if o.kind == Kind.PTERODACTYL_LOW:
                color = (210, 120, 40)        # laranja: agachar
            elif o.kind == Kind.PTERODACTYL:
                color = (120, 90, 160)        # roxo: aéreo alto
            else:
                color = (60, 120, 60)         # verde: cacto (pular)
            pygame.draw.rect(screen, color, self._irect(o.hitbox))
        score = self._font.render(f"score: {self.engine.score}  "
                                  f"vel: {self.engine.speed:.1f}", True, (83, 83, 83))
        screen.blit(score, (physics.WIDTH - 220, 12))
        pygame.display.flip()

    def _render_gameover(self, screen) -> None:
        if self.mode == MODE_AI:
            return
        msg = self._font.render("GAME OVER — ESPACO p/ reiniciar", True, (200, 40, 40))
        screen.blit(msg, (physics.WIDTH // 2 - 150, physics.HEIGHT // 2 - 10))
        pygame.display.flip()


if __name__ == "__main__":
    # Jogar no modo humano (ESPACO pula). Rodar de dentro de pc/ com:
    #   python -m game.dino_game
    DinoGame(mode=MODE_HUMAN).run_blocking()
