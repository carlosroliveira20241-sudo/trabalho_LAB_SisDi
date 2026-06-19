"""Aba Timers/ADC — temporizadores e conversor A/D (Atividade do PDF).

Timers: TCNT0/1/2 (contadores) e TCCRx (modo/prescaler).
ADC: ADMUX (canal/referência) e resultado (ADCH:ADCL, 10 bits).
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class TimersTab(Vertical):
    def __init__(self, monitor, **kwargs):
        super().__init__(**kwargs)
        self.monitor = monitor

    def compose(self) -> ComposeResult:
        yield Static("TIMERS & ADC", id="timers_title")
        yield Static("", id="timers_body")

    def on_mount(self) -> None:
        self.set_interval(0.3, self.refresh)

    def refresh(self) -> None:
        # TODO: monitor.read_timers() e monitor.read_adc(); montar a visualização.
        self.query_one("#timers_body", Static).update("(aguardando conexão serial)")
