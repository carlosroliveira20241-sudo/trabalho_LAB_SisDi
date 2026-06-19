"""NEAT — neuroevolução. O algoritmo clássico (e robusto) para o jogo do dino.

Cada genoma vira uma rede; é avaliado jogando um episódio headless; o fitness é
o tempo de sobrevivência (+ bônus por obstáculo passado). As melhores redes
sobrevivem, se reproduzem e sofrem mutação — sem gradiente, sem o colapso de
exploração do Policy Gradient.

Delegável ao Arduino: a avaliação de fitness de UM indivíduo (manda os pesos do
genoma, roda no Arduino, recebe o fitness) — útil pro benchmark medir quanto
custa avaliar um indivíduo no microcontrolador vs no PC. Por ora roda no PC.
"""
from __future__ import annotations

import os

import neat

from game.dino_game_headless import DinoGameHeadless, ACTION_JUMP, ACTION_DUCK, ACTION_NONE
from game import physics

MAX_FRAMES = 3000          # teto de duração de um episódio de avaliação
THRESHOLD = 0.0            # saída tanh > 0 ativa a ação
N_EVAL = 3                 # partidas por genoma (média reduz o ruído do fitness)

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "neat_config.txt")


def neat_inputs(game: DinoGameHeadless):
    """Estado de 4 entradas para o NEAT (mesmo `features()` da rede fixa):
    distância, velocidade e a posição vertical (topo e base) do próximo
    obstáculo — distingue cacto (pular) de pterodáctilo baixo (agachar)."""
    return game.features()


def decide(net, game: DinoGameHeadless) -> int:
    """Ativa a rede e escolhe entre nada / pular / agachar (2 saídas)."""
    jump_sig, duck_sig = net.activate(neat_inputs(game))
    if jump_sig > THRESHOLD and jump_sig >= duck_sig:
        return ACTION_JUMP
    if duck_sig > THRESHOLD:
        return ACTION_DUCK
    return ACTION_NONE


def play_genome(net, game: DinoGameHeadless) -> float:
    """Joga UM episódio e devolve o fitness (sobrevivência + bônus por obstáculo)."""
    while not game.is_dead and game.frames < MAX_FRAMES:
        game.step(decide(net, game))
    return float(game.frames + 10 * game.score)


def eval_genome(genome, config, kinds=None) -> float:
    """Fitness médio do genoma sobre N_EVAL partidas (reduz o ruído).

    `kinds` define o currículo (ex.: só cactos). None = todos os obstáculos.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total = 0.0
    for _ in range(N_EVAL):
        total += play_genome(net, DinoGameHeadless(kinds=kinds))
    return total / N_EVAL


class NeatTrainer:
    def __init__(self, config_path: str = DEFAULT_CONFIG, kinds=None):
        self.config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            config_path,
        )
        self.kinds = kinds          # currículo de obstáculos
        self.pop = neat.Population(self.config)
        self.stats = neat.StatisticsReporter()
        self.pop.add_reporter(neat.StdOutReporter(True))
        self.pop.add_reporter(self.stats)
        self.best = None

    def eval_population(self, genomes, config) -> None:
        """Avalia todos os genomas da geração (chamado pelo NEAT a cada geração)."""
        for _gid, genome in genomes:
            genome.fitness = eval_genome(genome, config, self.kinds)

    def run(self, generations: int = 50):
        """Roda a evolução por N gerações; devolve o melhor genoma."""
        self.best = self.pop.run(self.eval_population, generations)
        return self.best

    def run_one_generation(self):
        """Roda UMA geração e devolve o melhor genoma atual (para assistir)."""
        self.best = self.pop.run(self.eval_population, 1)
        return self.best
