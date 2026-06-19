"""Treino da IA no PC (Policy Gradient) — o dino aprende a jogar sozinho.

Tudo roda no PC; nenhum Arduino é necessário. As chamadas de `forward_pass`
passam pelo Router (localização = PC), então o benchmark já registra os tempos —
quando o Arduino entrar, é só delegar a função, sem mudar este script.

Uso (de dentro de pc/):
    python train_pc.py                 # assiste o dino aprendendo (janela pygame)
    python train_pc.py --fast          # treina sem janela, rápido, imprime progresso
    python train_pc.py --episodes 800  # nº de episódios
    python train_pc.py --fps 240       # velocidade da janela (modo assistir)
"""
from __future__ import annotations

import argparse

import numpy as np

from bench.benchmark import Benchmark
from comm import protocol
from comm.delegator import Router
from neural.neural_net import NeuralNet
from neural.policy_gradient import PolicyGradient
from game.dino_game_headless import DinoGameHeadless
from game import physics


def build_trainer():
    """Monta rede + router (PC-only) + Policy Gradient."""
    bench = Benchmark()
    router = Router(serial_comm=None, benchmark=bench)
    for name in list(router.location):       # sem Arduino: tudo no PC
        router.location[name] = protocol.LOC_PC
    net = NeuralNet(seed=0)
    router.register_pc("forward_pass", lambda s: net.forward(s))
    router.register_pc("backpropagation", lambda s, a, adv: net.backward(s, a, adv))
    router.register_pc("atualiza_pesos", lambda grad, lr: net.apply_gradients(grad, lr))
    trainer = PolicyGradient(router, net, lr=0.01)
    return trainer, net, bench


def run_episode(game: DinoGameHeadless, trainer: PolicyGradient, render=None) -> tuple[int, int]:
    game.reset()
    trainer._reset_episode()
    while not game.is_dead:
        state = game.features()
        action = trainer.act(state)          # 0 = nada, 1 = pular
        reward = game.step(action)
        trainer.observe(reward)
        if render is not None:
            render(game)
    trainer.end_episode()                    # atualiza os pesos
    return game.score, game.frames


class Renderer:
    """Janela pygame para assistir a IA jogando enquanto aprende."""

    def __init__(self, fps: int = 120):
        import pygame
        from game.obstacle import Kind
        self.pg = pygame
        self.Kind = Kind
        pygame.init()
        self.screen = pygame.display.set_mode((physics.WIDTH, physics.HEIGHT))
        pygame.display.set_caption("Dino Neural — IA treinando")
        self.font = pygame.font.SysFont("consolas", 18)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.episode = 0
        self.best = 0
        self.counter_label = "episodio"   # "geracao" no NEAT

    def draw(self, game: DinoGameHeadless) -> None:
        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.QUIT:
                raise KeyboardInterrupt
        self.screen.fill((247, 247, 247))
        pg.draw.line(self.screen, (83, 83, 83), (0, physics.GROUND_Y),
                     (physics.WIDTH, physics.GROUND_Y), 2)
        d = game.dino
        pg.draw.rect(self.screen, (83, 83, 83), [int(v) for v in d.hitbox])
        for o in game.obstacles:
            if o.kind == self.Kind.PTERODACTYL_LOW:
                color = (210, 120, 40)        # laranja: agachar
            elif o.kind == self.Kind.PTERODACTYL:
                color = (120, 90, 160)        # roxo: aéreo alto
            else:
                color = (60, 120, 60)         # verde: cacto (pular)
            pg.draw.rect(self.screen, color, [int(v) for v in o.hitbox])
        hud = self.font.render(
            f"{self.counter_label} {self.episode}   score {game.score}   melhor {self.best}",
            True, (83, 83, 83))
        self.screen.blit(hud, (12, 12))
        pg.display.flip()
        self.clock.tick(self.fps)


def main() -> None:
    ap = argparse.ArgumentParser(description="Treino PC do Dino Neural (Policy Gradient)")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=10, help="episódios por atualização")
    ap.add_argument("--fast", action="store_true", help="sem janela, máxima velocidade")
    ap.add_argument("--fps", type=int, default=120, help="velocidade da janela (assistir)")
    args = ap.parse_args()

    trainer, net, bench = build_trainer()
    game = DinoGameHeadless()
    scores: list[int] = []

    if args.fast:
        for ep in range(args.episodes):
            s, _ = run_episode(game, trainer)
            scores.append(s)
            if (ep + 1) % args.batch == 0:
                trainer.update()                  # atualiza a cada lote
            if ep % 50 == 0:
                avg = float(np.mean(scores[-50:]))
                print(f"ep {ep:4d}   score {s:3d}   melhor {max(scores):3d}   media50 {avg:5.1f}")
        print(f"\nfim — melhor score: {max(scores)}")
    else:
        r = Renderer(fps=args.fps)
        try:
            for ep in range(args.episodes):
                r.episode = ep
                s, _ = run_episode(game, trainer, render=r.draw)
                scores.append(s)
                r.best = max(r.best, s)
                if (ep + 1) % args.batch == 0:
                    trainer.update()
        except KeyboardInterrupt:
            pass
        if scores:
            print(f"melhor score: {max(scores)}  em {len(scores)} episodios")


if __name__ == "__main__":
    main()
