"""Coletor de estatísticas do benchmark de delegação.

Para cada função delegável, mantém janelas rolantes de tempos por localização
(PC / Arduino), permitindo exibir em tempo real: último tempo, média, p95,
chamadas/s e contadores acumulados. Thread-safe (o Router escreve de uma thread,
a UI lê de outra).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

WINDOW = 128  # nº de amostras na janela rolante


@dataclass
class Stat:
    """Estatísticas de uma (função, localização)."""
    t_compute: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    t_roundtrip: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    total_calls: int = 0

    def mean_compute(self) -> float:
        return sum(self.t_compute) / len(self.t_compute) if self.t_compute else 0.0

    def mean_roundtrip(self) -> float:
        return sum(self.t_roundtrip) / len(self.t_roundtrip) if self.t_roundtrip else 0.0

    def mean_serial(self) -> float:
        """Overhead de transporte = round-trip - compute (médias)."""
        return max(0.0, self.mean_roundtrip() - self.mean_compute())

    def p95_roundtrip(self) -> float:
        if not self.t_roundtrip:
            return 0.0
        s = sorted(self.t_roundtrip)
        return s[min(len(s) - 1, int(0.95 * len(s)))]

    def calls_per_sec(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        span = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / span if span > 0 else 0.0

    def last_compute(self) -> float:
        return self.t_compute[-1] if self.t_compute else 0.0


class Benchmark:
    def __init__(self):
        self._lock = threading.Lock()
        # chave: (func_name, location) -> Stat
        self._stats: dict[tuple[str, int], Stat] = {}

    def record(self, name: str, location: int, t_compute: float, t_roundtrip: float) -> None:
        key = (name, location)
        with self._lock:
            st = self._stats.setdefault(key, Stat())
            st.t_compute.append(t_compute)
            st.t_roundtrip.append(t_roundtrip)
            st.timestamps.append(time.perf_counter())
            st.total_calls += 1

    def get(self, name: str, location: int) -> Stat:
        with self._lock:
            return self._stats.get((name, location), Stat())

    def snapshot(self) -> dict[tuple[str, int], Stat]:
        """Cópia rasa para a UI ler sem segurar o lock por muito tempo."""
        with self._lock:
            return dict(self._stats)
