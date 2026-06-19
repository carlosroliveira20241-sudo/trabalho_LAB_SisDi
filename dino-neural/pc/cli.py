"""Modo terminal (opção 2 do PDF) — dashboard via rich, sem abas.

Roda quando `main.py --cli`. Mostra tudo num único Live layout: estado do jogo,
benchmark de delegação e um resumo do monitor (portas, flags, ADC). Mais simples
que o Textual, atende ao requisito de "via terminal".
"""
from __future__ import annotations

import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.table import Table

from comm import protocol


def build_bench_table(router, benchmark) -> Table:
    t = Table(title="Delegação & Benchmark")
    t.add_column("função"); t.add_column("local")
    t.add_column("compute"); t.add_column("serial"); t.add_column("round-trip")
    for name in ("forward_pass", "backpropagation", "atualiza_pesos", "calcula_recompensa"):
        loc = router.location.get(name, protocol.LOC_PC)
        st = benchmark.get(name, loc)
        t.add_row(
            name, "ARDUINO" if loc == protocol.LOC_ARDUINO else "PC",
            f"{st.last_compute()*1e6:.0f}µs",
            f"{st.mean_serial()*1e3:.2f}ms",
            f"{st.mean_roundtrip()*1e3:.2f}ms",
        )
    return t


def run_cli(game, router, benchmark, monitor) -> None:
    console = Console()
    layout = Layout()
    with Live(layout, console=console, refresh_per_second=4):
        while True:
            layout.update(build_bench_table(router, benchmark))
            # TODO: adicionar painéis de jogo e monitor ao layout
            time.sleep(0.25)
