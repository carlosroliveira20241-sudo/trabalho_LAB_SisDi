"""Router de delegação: o coração do projeto.

Cada função delegável tem DUAS implementações com a mesma assinatura:
  - uma local (PC, numpy)
  - uma remota (Arduino, via SerialComm)

O Router guarda um mapa func_id -> localização e despacha cada chamada para o
lado ativo, cronometrando a execução e registrando no Benchmark. Trocar de lado
("hot-swap") é mudar uma entrada do mapa — mas antes é preciso SINCRONIZAR O
ESTADO (ex.: pesos da rede) para que a migração seja transparente para a IA.

Fluxo de migração (move):
  1. pausa chamadas da função (lock)
  2. lê o estado do lado atual (ex.: pesos)
  3. escreve o estado no lado destino
  4. troca a localização no mapa
  5. libera o lock -> próximas chamadas já vão pro novo lado
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from . import protocol
from .serial_comm import SerialComm
from bench.benchmark import Benchmark

# nomes legíveis <-> func_id do protocolo
FUNC_IDS = {
    "forward_pass":      protocol.FUNC_FORWARD,
    "backpropagation":   protocol.FUNC_BACKPROP,
    "atualiza_pesos":    protocol.FUNC_UPDATE_W,
    "extrai_estado":     protocol.FUNC_EXTRACT,
    "calcula_recompensa": protocol.FUNC_REWARD,
}

# padrão de localização inicial (ver tabela do blueprint)
DEFAULT_LOCATION = {
    "forward_pass":      protocol.LOC_ARDUINO,
    "backpropagation":   protocol.LOC_ARDUINO,
    "atualiza_pesos":    protocol.LOC_ARDUINO,
    "extrai_estado":     protocol.LOC_PC,
    "calcula_recompensa": protocol.LOC_PC,
}

# extrai_estado e envia_acao são sempre no PC (não delegáveis) -> não migram.
NON_DELEGABLE = {"envia_acao_ao_jogo"}


class Router:
    def __init__(self, serial_comm: SerialComm, benchmark: Benchmark):
        self.serial = serial_comm
        self.bench = benchmark
        self.location = dict(DEFAULT_LOCATION)
        self._pc_impls: dict[str, Callable] = {}
        self._arduino_cmds: dict[str, int] = {
            "forward_pass": protocol.CMD_FORWARD,
            "backpropagation": protocol.CMD_BACKPROP,
            "calcula_recompensa": protocol.CMD_REWARD,
        }
        self._locks: dict[str, threading.Lock] = {
            name: threading.Lock() for name in FUNC_IDS
        }

    # --- registro das implementações de PC ----------------------------------
    def register_pc(self, name: str, fn: Callable) -> None:
        """Associa a implementação local (numpy) de uma função."""
        self._pc_impls[name] = fn

    # --- despacho ------------------------------------------------------------
    def call(self, name: str, *args, **kwargs):
        """Roteia a chamada para o lado ativo, mede o tempo e registra."""
        with self._locks[name]:
            loc = self.location[name]
            if loc == protocol.LOC_PC:
                return self._call_pc(name, *args, **kwargs)
            return self._call_arduino(name, *args, **kwargs)

    def _call_pc(self, name: str, *args, **kwargs):
        fn = self._pc_impls[name]
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        t_compute = time.perf_counter() - t0
        self.bench.record(name, protocol.LOC_PC, t_compute=t_compute, t_roundtrip=t_compute)
        return result

    def _call_arduino(self, name: str, *args, **kwargs):
        # TODO: empacotar args conforme o comando (ex.: forward = 3 floats)
        cmd = self._arduino_cmds[name]
        payload = self._pack_args(name, *args, **kwargs)
        resp = self.serial.request(cmd, payload)
        # resposta começa com u32 t_compute (micros) -> ver protocolo
        t_compute = protocol.unpack_u32(resp.payload, 0) / 1e6
        result = self._unpack_result(name, resp.payload[4:])
        self.bench.record(
            name, protocol.LOC_ARDUINO,
            t_compute=t_compute, t_roundtrip=resp.t_roundtrip,
        )
        return result

    # --- hot-swap ------------------------------------------------------------
    def move(self, name: str, to: int, sync_state: Callable | None = None) -> None:
        """Migra uma função para PC (0) ou Arduino (1), com sync de estado.

        `sync_state(from_loc, to_loc)` é chamado com o lock segurado, antes da
        troca, para sincronizar pesos/estado entre os lados.
        """
        if name in NON_DELEGABLE:
            raise ValueError(f"{name} não é delegável")
        with self._locks[name]:
            frm = self.location[name]
            if frm == to:
                return
            if sync_state is not None:
                sync_state(frm, to)
            # avisa o Arduino sobre a nova localização
            self.serial.request(
                protocol.CMD_DELEGATE,
                bytes([FUNC_IDS[name], to]),
            )
            self.location[name] = to

    # --- helpers de (de)serialização por função -----------------------------
    def _pack_args(self, name: str, *args, **kwargs) -> bytes:
        if name == "forward_pass":
            state = args[0]                       # [dist, vel, altura]
            return protocol.pack_floats(*[float(x) for x in state])
        if name == "calcula_recompensa":
            passed, died = args[0], args[1]       # bool/int
            return bytes([1 if passed else 0, 1 if died else 0])
        raise NotImplementedError(f"empacotamento de {name} ainda não implementado")

    def _unpack_result(self, name: str, data: bytes):
        import numpy as np
        if name == "forward_pass":
            return np.asarray(protocol.unpack_floats(data))   # [out_0, out_1]
        if name == "calcula_recompensa":
            return protocol.unpack_floats(data)[0]            # f32 reward
        raise NotImplementedError(f"desempacotamento de {name} ainda não implementado")
