"""App Textual com abas — interface interativa (opção 1 do PDF).

Junta as telas: Jogo, Treino, Delegação/Benchmark, Memória, Regs/Flags, Timers/ADC.
Recebe os objetos já construídos (game, router, benchmark, monitor, trainer_ref)
do main.py, que cuida do serial e das threads.
"""
from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, TabbedContent, TabPane

from ui.tab_game import GameTab
from ui.tab_train import TrainTab
from ui.tab_delegate import DelegateTab
from ui.tab_memory import MemoryTab
from ui.tab_regs import RegsTab
from ui.tab_timers import TimersTab


class DinoNeuralApp(App):
    CSS = """
    DataTable { height: auto; }
    """

    BINDINGS = [
        ("q", "quit", "sair"),
        # migração de funções (aba Delegação)
        ("f", "toggle('forward_pass')", "swap forward"),
        ("b", "toggle('backpropagation')", "swap backprop"),
        ("w", "toggle('atualiza_pesos')", "swap pesos"),
        ("r", "toggle('calcula_recompensa')", "swap reward"),
        # troca de algoritmo (aba Treino)
        ("1", "algo(0)", "Policy Gradient"),
        ("2", "algo(1)", "DQN"),
        ("3", "algo(2)", "NEAT"),
    ]

    def __init__(self, game, router, benchmark, monitor, trainer_ref):
        super().__init__()
        self.game = game
        self.router = router
        self.bench = benchmark
        self.monitor = monitor
        self.trainer_ref = trainer_ref

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Jogo", id="tab-game"):
                yield GameTab(self.game)
            with TabPane("Treino", id="tab-train"):
                yield TrainTab(self.trainer_ref)
            with TabPane("Delegação", id="tab-delegate"):
                yield DelegateTab(self.router, self.bench)
            with TabPane("Memória", id="tab-memory"):
                yield MemoryTab(self.monitor)
            with TabPane("Regs/Flags", id="tab-regs"):
                yield RegsTab(self.monitor)
            with TabPane("Timers/ADC", id="tab-timers"):
                yield TimersTab(self.monitor)
        yield Footer()

    # --- ações de teclado ---------------------------------------------------
    def action_toggle(self, name: str) -> None:
        self.query_one(DelegateTab).toggle(name)

    def action_algo(self, algo_id: int) -> None:
        self.query_one(TrainTab).set_algorithm(algo_id)
