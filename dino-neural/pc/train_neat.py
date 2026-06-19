"""Treino por NEAT — assista a população evoluir e ficar boa no dino.

Currículo: por padrão treina SÓ com cactos (tarefa sem conflito, onde o NEAT
fica excelente — score nas dezenas/centenas). Com --pteros inclui pterodáctilos
(bem mais difícil: a rede precisa NÃO pular neles, discriminando pela altura).

Uso (de dentro de pc/):
    python train_neat.py                 # assiste a evolução (janela), só cactos
    python train_neat.py --fast          # sem janela, evolui rápido e imprime
    python train_neat.py --pteros        # inclui pterodáctilos (difícil)
    python train_neat.py --generations 80 --fps 200
"""
from __future__ import annotations

import argparse
import pickle

import neat

from neural.neat_trainer import NeatTrainer, decide, MAX_FRAMES
from game.dino_game_headless import DinoGameHeadless
from game.obstacle import CACTUS_ONLY, CACTUS_AND_DUCK, ALL_KINDS

WINNER_PATH = "neat_winner.pkl"


def watch_genome(genome, config, renderer, kinds) -> None:
    """Renderiza UMA partida do melhor genoma da geração (modo assistir)."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = DinoGameHeadless(kinds=kinds)
    while not game.is_dead and game.frames < MAX_FRAMES:
        game.step(decide(net, game))
        renderer.draw(game)


def main() -> None:
    ap = argparse.ArgumentParser(description="Treino NEAT do Dino Neural")
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--duck", action="store_true", help="cactos + obstáculo de agachar")
    ap.add_argument("--pteros", action="store_true", help="todos os obstáculos (difícil)")
    ap.add_argument("--fast", action="store_true", help="sem janela, evolui rápido")
    ap.add_argument("--fps", type=int, default=200, help="velocidade da janela")
    args = ap.parse_args()

    if args.pteros:
        kinds, label = ALL_KINDS, "todos os obstáculos"
    elif args.duck:
        kinds, label = CACTUS_AND_DUCK, "cactos + agachar"
    else:
        kinds, label = CACTUS_ONLY, "só cactos"
    trainer = NeatTrainer(kinds=kinds)
    print(f"currículo: {label}")

    if args.fast:
        winner = trainer.run(args.generations)
        print(f"\nmelhor fitness: {round(winner.fitness)}")
    else:
        from train_pc import Renderer
        r = Renderer(fps=args.fps)
        r.counter_label = "geracao"
        winner = None
        try:
            for gen in range(args.generations):
                winner = trainer.run_one_generation()   # evolui 1 geração
                r.episode = gen
                r.best = round(winner.fitness)
                watch_genome(winner, trainer.config, r, kinds)  # mostra o campeão jogando
        except KeyboardInterrupt:
            pass

    if winner is not None:
        with open(WINNER_PATH, "wb") as f:
            pickle.dump(winner, f)
        print(f"campeão salvo em {WINNER_PATH}")


if __name__ == "__main__":
    main()
