"""Bootstrap do Dino Neural: monta serial, threads e UI.

Uso:
    python main.py                 # interface com abas (Textual)
    python main.py --cli           # modo terminal (rich)
    python main.py --port COM3     # escolhe a porta serial
    python main.py --no-serial     # roda só no PC (sem Arduino, p/ testar a UI)

Orquestração (ver README):
  - thread do jogo: loop do Pygame, FPS livre, desacoplado do serial
  - thread da IA  : extrai estado -> router.call(forward) -> ação -> reward -> treino
  - thread serial : dentro do SerialComm (1 transação por vez)
  - thread da UI  : Textual/rich, lê snapshots do benchmark e do monitor
"""
from __future__ import annotations

import argparse
import threading
import time

from comm.serial_comm import SerialComm
from comm.delegator import Router
from comm.monitor import Monitor
from bench.benchmark import Benchmark
from game.dino_game import DinoGame
from game.dino_game_headless import ACTION_NONE
from neural.neural_net import NeuralNet
from neural.policy_gradient import PolicyGradient


def build_pc_impls(router: Router, net: NeuralNet) -> None:
    """Registra as implementações locais (PC) das funções delegáveis."""
    router.register_pc("forward_pass", lambda state: net.forward(state))
    # TODO: backpropagation, atualiza_pesos, calcula_recompensa
    #       (cada uma com a mesma assinatura da versão Arduino)


def ai_loop(game: DinoGame, trainer, stop: threading.Event) -> None:
    """Decide ações e treina. Ritmo limitado pelo lado ativo (vide benchmark)."""
    while not stop.is_set():
        state = game.engine.state_vector()
        try:
            action = trainer.act(state)
        except NotImplementedError:
            action = ACTION_NONE
        game.set_action(action)
        # TODO: observar recompensa, detectar fim de episódio -> trainer.end_episode()
        time.sleep(0.0)  # cede a CPU; o gargalo real é o forward (PC ou serial)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dino Neural")
    parser.add_argument("--port", default=None, help="porta serial (ex.: COM3)")
    parser.add_argument("--baud", type=int, default=250000)
    parser.add_argument("--cli", action="store_true", help="modo terminal (rich)")
    parser.add_argument("--no-serial", action="store_true", help="roda só no PC")
    args = parser.parse_args()

    # --- núcleo ------------------------------------------------------------
    benchmark = Benchmark()
    serial_comm = SerialComm(args.port, args.baud) if (args.port and not args.no_serial) else None
    if serial_comm:
        serial_comm.open()
    router = Router(serial_comm, benchmark)
    monitor = Monitor(serial_comm) if serial_comm else None

    net = NeuralNet(seed=0)
    build_pc_impls(router, net)

    # se não há serial, força tudo no PC (não há para onde delegar)
    if serial_comm is None:
        from comm import protocol
        for name in list(router.location):
            router.location[name] = protocol.LOC_PC

    trainer = PolicyGradient(router, net)
    trainer_ref = {"active": trainer}  # mutável p/ a aba Treino trocar de algoritmo

    # --- threads -----------------------------------------------------------
    game = DinoGame()
    game.start()
    stop = threading.Event()
    threading.Thread(target=ai_loop, args=(game, trainer, stop), daemon=True).start()

    # --- UI ----------------------------------------------------------------
    try:
        if args.cli:
            from cli import run_cli
            run_cli(game, router, benchmark, monitor)
        else:
            from app import DinoNeuralApp
            DinoNeuralApp(game, router, benchmark, monitor, trainer_ref).run()
    finally:
        stop.set()
        game.stop()
        if serial_comm:
            serial_comm.close()


if __name__ == "__main__":
    main()
