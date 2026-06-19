"""Aba Jogo — estado do jogo em tempo real (o render gráfico é janela Pygame)."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class GameTab(Vertical):
    def __init__(self, game, **kwargs):
        super().__init__(**kwargs)
        self.game = game

    def compose(self) -> ComposeResult:
        yield Static("JOGO", id="game_title")
        yield Static("", id="game_state")

    def on_mount(self) -> None:
        self.set_interval(0.1, self.refresh_state)

    def refresh_state(self) -> None:
        s = self.game.game_state()
        text = (
            f"score: {s['score']}\n"
            f"dist obstáculo : {s['distance_to_obstacle']:.1f}\n"
            f"altura obstáculo: {s['obstacle_height']:.1f}\n"
            f"velocidade     : {s['current_speed']:.2f}\n"
            f"dino_y         : {s['dino_y']:.1f}\n"
            f"morto          : {s['is_dead']}"
        )
        self.query_one("#game_state", Static).update(text)
        # TODO: seletor de modo (humano/IA/híbrido/NEAT)
