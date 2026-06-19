"""Camada de transporte serial: abre a porta, envia frames e lê respostas.

Roda em thread própria. O resto do app fala com o Arduino exclusivamente por
aqui, via `request()` (envia um frame e espera a resposta correspondente).

Esta classe NÃO conhece a semântica dos comandos — só o framing. Quem interpreta
payloads é o Router (comandos neurais) e o Monitor (comandos de dump).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import serial  # pyserial

from . import protocol


@dataclass
class Response:
    cmd: int
    payload: bytes
    t_roundtrip: float  # segundos, wall-clock send->recv (medido no PC)


class SerialComm:
    def __init__(self, port: str, baud: int = 250000, timeout: float = 1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser: serial.Serial | None = None
        self._lock = threading.Lock()  # serializa request/response (1 em voo)

    # --- conexão ------------------------------------------------------------
    def open(self) -> None:
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        time.sleep(2.0)  # Uno reseta ao abrir a serial; espera o bootloader
        self._ser.reset_input_buffer()

    def close(self) -> None:
        if self._ser:
            self._ser.close()
            self._ser = None

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # --- transação ----------------------------------------------------------
    def request(self, cmd: int, payload: bytes = b"") -> Response:
        """Envia um frame e bloqueia até receber a resposta (mesmo cmd).

        Retorna a resposta com o round-trip medido no PC. Lança em timeout
        ou checksum inválido.
        """
        if not self.is_open:
            raise RuntimeError("serial não está aberta")
        frame = protocol.encode(cmd, payload)
        with self._lock:
            t0 = time.perf_counter()
            try:
                self._ser.write(frame)
                resp_cmd, resp_payload = self._read_frame()
            except Exception:
                # pacote corrompido/timeout: limpa o buffer p/ a próxima ficar sã
                try:
                    self._ser.reset_input_buffer()
                except Exception:
                    pass
                raise
            t_rt = time.perf_counter() - t0
        return Response(resp_cmd, resp_payload, t_rt)

    # --- leitura de frame (máquina de estados) ------------------------------
    def _read_frame(self) -> tuple[int, bytes]:
        """Lê um frame [0xAA][cmd][len][payload][chk], ressincronizando no START.

        Bloqueia até um frame completo chegar (respeitando o timeout da porta).
        Lança TimeoutError se a resposta não vier, ou ValueError em checksum ruim.
        """
        # 1. procura o byte de START (descarta lixo até achar)
        while True:
            b = self._ser.read(1)
            if not b:
                raise TimeoutError("timeout esperando START do Arduino")
            if b[0] == protocol.START:
                break
        # 2. cabeçalho: cmd + len
        header = self._ser.read(2)
        if len(header) < 2:
            raise TimeoutError("timeout lendo cabeçalho")
        cmd, length = header[0], header[1]
        # 3. payload
        payload = self._ser.read(length) if length else b""
        if len(payload) < length:
            raise TimeoutError("timeout lendo payload")
        # 4. checksum
        chk = self._ser.read(1)
        if len(chk) < 1:
            raise TimeoutError("timeout lendo checksum")
        if chk[0] != protocol.checksum(cmd, payload):
            raise ValueError("checksum inválido (dados corrompidos na serial)")
        return cmd, payload
