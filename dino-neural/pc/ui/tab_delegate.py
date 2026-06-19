"""Aba Delegação & Benchmark — a tela principal do projeto.

Mostra, por função delegável e em tempo real:
  localização (PC/Arduino), t_compute, t_serial, round-trip, calls/s e sparkline.
Teclas migram a função de lado SEM parar a IA; os números se reescrevem ao vivo
e a aba Memória passa a refletir a rede mudando na SRAM quando o trabalho está
no Arduino.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from comm import protocol

# linhas da tabela (funções delegáveis) e a tecla que migra cada uma
ROWS = [
    ("forward_pass", "F"),
    ("backpropagation", "B"),
    ("atualiza_pesos", "W"),
    ("calcula_recompensa", "R"),
]


def fmt_time(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    return f"{seconds * 1e3:.2f} ms"


def sparkline(values, width: int = 8) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    if not values:
        return " " * width
    vals = list(values)[-width:]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(blocks[min(7, int((v - lo) / rng * 7))] for v in vals)


class DelegateTab(Vertical):
    """Recebe `router` e `benchmark` na construção; faz refresh periódico."""

    def __init__(self, router, benchmark, **kwargs):
        super().__init__(**kwargs)
        self.router = router
        self.bench = benchmark

    def compose(self) -> ComposeResult:
        yield Static("DELEGAÇÃO & BENCHMARK — [F/B/W/R] migram a função", id="title")
        yield DataTable(id="bench_table")
        yield Static("", id="totals")

    def on_mount(self) -> None:
        table = self.query_one("#bench_table", DataTable)
        table.add_columns("função", "local", "t_compute", "t_serial",
                          "round-trip", "calls/s", "histórico")
        self.set_interval(0.25, self.refresh_table)  # 4 Hz

    def refresh_table(self) -> None:
        table = self.query_one("#bench_table", DataTable)
        table.clear()
        for name, _key in ROWS:
            loc = self.router.location.get(name, protocol.LOC_PC)
            st = self.bench.get(name, loc)
            loc_label = "ARDUINO" if loc == protocol.LOC_ARDUINO else "PC"
            table.add_row(
                name, loc_label,
                fmt_time(st.last_compute()),
                fmt_time(st.mean_serial()),
                fmt_time(st.mean_roundtrip()),
                f"{st.calls_per_sec():.0f}",
                sparkline(st.t_roundtrip),
            )
        # TODO: linha de acumulados (total no PC vs total no Arduino)

    # --- migração por tecla (ligada em app.py via BINDINGS) -----------------
    def toggle(self, name: str) -> None:
        cur = self.router.location.get(name, protocol.LOC_PC)
        new = protocol.LOC_PC if cur == protocol.LOC_ARDUINO else protocol.LOC_ARDUINO
        # TODO: passar sync_state real (ler pesos de um lado, escrever no outro)
        self.router.move(name, to=new)
