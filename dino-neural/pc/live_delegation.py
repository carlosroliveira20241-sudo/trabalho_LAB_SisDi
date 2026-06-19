"""Delegação dinâmica ao vivo — o clímax do projeto.

O dino joga sozinho usando a rede fixa 3->4->2. A cada frame, o `forward_pass`
que decide a ação passa pelo Router, que pode executá-lo no PC ou no Arduino.
Você aperta [D] e a função MIGRA de lado em tempo real, sem parar o jogo — e o
painel mostra na hora o tempo de cálculo, o overhead do serial e as chamadas/s
de cada lado. É a demonstração de "trocar quem faz o quê enquanto a IA roda".

Os pesos são sincronizados nos dois lados no início, então o PC e o Arduino
calculam IGUAL — só muda ONDE o cálculo roda (e quanto custa).

Uso (de dentro de pc/):
    python live_delegation.py                 # escolhe a porta no terminal
    python live_delegation.py --port COM4
    python live_delegation.py --fps 60
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pygame

from comm import protocol
from comm.serial_comm import SerialComm
from comm.port_select import choose_port
from comm.delegator import Router
from bench.benchmark import Benchmark
from neural.neural_net import NeuralNet, dino_policy_weights
from game.dino_game_headless import DinoGameHeadless, ACTION_JUMP, ACTION_DUCK, ACTION_NONE
from game.obstacle import CACTUS_AND_DUCK, Kind
from game import physics

LOC_NAME = {protocol.LOC_PC: "PC", protocol.LOC_ARDUINO: "ARDUINO"}
LOC_COLOR = {protocol.LOC_PC: (40, 130, 200), protocol.LOC_ARDUINO: (210, 120, 40)}


def fmt(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    return f"{seconds * 1e3:.2f} ms"


def main() -> None:
    ap = argparse.ArgumentParser(description="Delegação dinâmica ao vivo PC<->Arduino")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=250000)
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()

    port = choose_port(args.port)
    if not port:
        print("Nenhuma porta selecionada.")
        return

    print(f"Conectando em {port}...")
    sc = SerialComm(port, args.baud)
    sc.open()

    # checa a conexão e sincroniza os pesos calibrados nos dois lados
    if sc.request(protocol.CMD_PING, b"\x01").payload != b"\x01":
        print("PING falhou — o sketch certo está na placa?")
        sc.close()
        return
    net = NeuralNet()
    net.set_weights(dino_policy_weights())
    sc.request(protocol.CMD_SET_WEIGHTS, protocol.pack_floats(*net.get_weights()))
    print("pesos sincronizados PC<->Arduino. Abrindo o jogo...")

    bench = Benchmark()
    router = Router(sc, bench)
    router.register_pc("forward_pass", lambda s: net.forward(s))
    router.location["forward_pass"] = protocol.LOC_PC   # começa no PC

    pygame.init()
    screen = pygame.display.set_mode((physics.WIDTH, physics.HEIGHT + 90))
    pygame.display.set_caption("Dino Neural — Delegação ao vivo")
    font = pygame.font.SysFont("consolas", 16)
    big = pygame.font.SysFont("consolas", 22, bold=True)
    clock = pygame.time.Clock()

    game = DinoGameHeadless(kinds=CACTUS_AND_DUCK)
    serial_warn = ""
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_d, pygame.K_SPACE):
                    cur = router.location["forward_pass"]
                    router.location["forward_pass"] = (
                        protocol.LOC_ARDUINO if cur == protocol.LOC_PC else protocol.LOC_PC
                    )
                elif e.key == pygame.K_ESCAPE:
                    running = False

        loc = router.location["forward_pass"]
        try:
            probs = router.call("forward_pass", game.features())
            serial_warn = ""
        except Exception as ex:          # erro no serial: cai pro PC e avisa
            router.location["forward_pass"] = protocol.LOC_PC
            serial_warn = f"serial falhou ({ex}); voltando ao PC"
            probs = net.forward(game.features())
            loc = protocol.LOC_PC

        game.step(int(np.argmax(probs)))   # 0 nada, 1 pular, 2 abaixar
        if game.is_dead:
            game.reset()

        _render(screen, font, big, game, router, bench, clock, serial_warn)
        clock.tick(args.fps)

    sc.close()
    pygame.quit()


def _render(screen, font, big, game, router, bench, clock, warn) -> None:
    loc = router.location["forward_pass"]
    st = bench.get("forward_pass", loc)

    # --- jogo (parte de cima) ---
    screen.fill((247, 247, 247))
    pygame.draw.line(screen, (83, 83, 83), (0, physics.GROUND_Y),
                     (physics.WIDTH, physics.GROUND_Y), 2)
    pygame.draw.rect(screen, (83, 83, 83), [int(v) for v in game.dino.hitbox])
    for o in game.obstacles:
        c = (210, 120, 40) if o.kind == Kind.PTERODACTYL_LOW else (60, 120, 60)
        pygame.draw.rect(screen, c, [int(v) for v in o.hitbox])
    screen.blit(font.render(f"score {game.score}", True, (83, 83, 83)),
                (physics.WIDTH - 110, 12))

    # --- painel de delegação (parte de baixo) ---
    panel_y = physics.HEIGHT + 6
    pygame.draw.rect(screen, (28, 28, 32), (0, physics.HEIGHT, physics.WIDTH, 90))

    label = big.render(f"forward_pass  @  {LOC_NAME[loc]}", True, LOC_COLOR[loc])
    screen.blit(label, (12, panel_y))

    cols = [
        ("compute", fmt(st.last_compute())),
        ("serial", fmt(st.mean_serial()) if loc == protocol.LOC_ARDUINO else "—"),
        ("round-trip", fmt(st.mean_roundtrip())),
        ("chamadas/s", f"{st.calls_per_sec():.0f}"),
        ("FPS", f"{clock.get_fps():.0f}"),
    ]
    x = 360
    for name, val in cols:
        screen.blit(font.render(name, True, (150, 150, 150)), (x, panel_y))
        screen.blit(font.render(val, True, (235, 235, 235)), (x, panel_y + 22))
        x += 90

    hint = "[D] ou [ESPACO] trocar PC <-> Arduino     [ESC] sair"
    screen.blit(font.render(hint, True, (140, 140, 140)), (12, panel_y + 52))
    if warn:
        screen.blit(font.render(warn, True, (230, 90, 90)), (12, panel_y + 70))

    pygame.display.flip()


if __name__ == "__main__":
    main()
