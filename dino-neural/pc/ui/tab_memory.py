"""Aba Memória — dump em tempo real de SRAM/FLASH/EEPROM (Atividade do PDF).

Destaque: a região da SRAM onde a rede neural vive é realçada, então dá pra ver
os pesos mudando enquanto a IA treina (ainda mais quando forward/backprop estão
delegados ao Arduino).
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

BYTES_PER_LINE = 16


def hexdump(data: bytes, base_addr: int = 0, highlight: range | None = None) -> str:
    lines = []
    for off in range(0, len(data), BYTES_PER_LINE):
        chunk = data[off:off + BYTES_PER_LINE]
        addr = base_addr + off
        hexs = " ".join(f"{b:02X}" for b in chunk)
        ascii_ = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{addr:04X}  {hexs:<47}  {ascii_}")
    # TODO: aplicar realce (cor) na faixa `highlight` (região dos pesos)
    return "\n".join(lines)


class MemoryTab(Vertical):
    def __init__(self, monitor, **kwargs):
        super().__init__(**kwargs)
        self.monitor = monitor
        self.region = "SRAM"  # SRAM | EEPROM | FLASH
        self.addr = 0x0100

    def compose(self) -> ComposeResult:
        yield Static("MEMÓRIA — [S]RAM  [E]EPROM  [L]FLASH   ←/→ navega", id="mem_title")
        yield Static("", id="mem_dump")

    def on_mount(self) -> None:
        self.set_interval(0.5, self.refresh_dump)

    def refresh_dump(self) -> None:
        # TODO: ler o bloco atual via monitor.read_sram/eeprom/flash e desenhar
        #       com realce da região dos pesos (monitor.net_addr / net_size).
        self.query_one("#mem_dump", Static).update("(aguardando conexão serial)")
