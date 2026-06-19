"""Aba Treino — algoritmo ativo, recompensa, episódio e curva de aprendizado."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from comm import protocol

ALGO_NAMES = {protocol.ALGO_PG: "Policy Gradient",
              protocol.ALGO_DQN: "DQN",
              protocol.ALGO_NEAT: "NEAT"}


class TrainTab(Vertical):
    def __init__(self, trainer_ref, **kwargs):
        super().__init__(**kwargs)
        self.trainer_ref = trainer_ref  # referência mutável ao trainer ativo

    def compose(self) -> ComposeResult:
        yield Static("TREINO", id="train_title")
        yield Static("", id="train_state")

    def on_mount(self) -> None:
        self.set_interval(0.25, self.refresh_state)

    def refresh_state(self) -> None:
        # TODO: ler episódio, recompensa acumulada, melhor score, epsilon (DQN),
        #       geração/fitness (NEAT); desenhar curva com sparkline.
        self.query_one("#train_state", Static).update("(em construção)")

    # tecla 1/2/3 troca o algoritmo (ligado em app.py)
    def set_algorithm(self, algo_id: int) -> None:
        # TODO: trocar o trainer ativo e mandar CMD_SET_ALGO ao Arduino
        ...
