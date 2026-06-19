"""Aba Registradores/Flags — portas digitais e SREG bit a bit (Atividade do PDF).

Mostra PORTB/C/D com DDRx (direção) e PINx (leitura), cada bit visualmente, e as
flags do SREG (I T H S V N Z C).
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


def bits(value: int, width: int = 8) -> str:
    """Representação visual: bit 1 = ●, bit 0 = ·, do MSB ao LSB."""
    return " ".join("●" if (value >> (width - 1 - i)) & 1 else "·" for i in range(width))


class RegsTab(Vertical):
    def __init__(self, monitor, **kwargs):
        super().__init__(**kwargs)
        self.monitor = monitor

    def compose(self) -> ComposeResult:
        yield Static("PORTAS & FLAGS", id="regs_title")
        yield Static("", id="regs_body")

    def on_mount(self) -> None:
        self.set_interval(0.3, self.refresh_regs)

    def refresh_regs(self) -> None:
        # TODO: ports = monitor.read_ports(); flags = monitor.read_flags()
        #       montar tabela: PORTB/C/D + DDR + PIN com bits(); SREG com nomes.
        self.query_one("#regs_body", Static).update("(aguardando conexão serial)")
