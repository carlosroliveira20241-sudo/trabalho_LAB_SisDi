"""Protocolo serial PC <-> Arduino (lado PC).

Framing:  [0xAA] [cmd] [len] [payload... (len bytes)] [checksum]
  - 0xAA  : byte de sincronismo (START)
  - cmd   : código do comando (ver constantes abaixo)
  - len   : tamanho do payload em bytes (0..255)
  - payload: dados do comando
  - checksum: XOR de (cmd ^ len ^ payload[*]) para detecção de erro

Floats trafegam como IEEE-754 32 bits little-endian (struct '<f'),
que é o layout nativo do AVR-GCC no ATmega328P.

IMPORTANTE: este arquivo é o espelho de `arduino/protocol.h`. Qualquer mudança
em um código de comando precisa ser refletida no outro.
"""
from __future__ import annotations

import struct

START = 0xAA

CMD_PING          = 0x00   # teste de conexão: o Arduino ecoa o payload de volta

# --- PC -> Arduino : comandos NEURAIS ---------------------------------------
CMD_FORWARD       = 0x01   # payload: f32 dist, f32 vel, f32 altura
CMD_BACKPROP      = 0x02   # payload: f32 dist, f32 vel, f32 altura, f32 reward
CMD_DELEGATE      = 0x03   # payload: u8 func_id, u8 location (0=PC, 1=Arduino)
CMD_GET_WEIGHTS   = 0x04   # payload: vazio
CMD_SET_ALGO      = 0x05   # payload: u8 algo_id (0=PG, 1=DQN, 2=NEAT)
CMD_SET_WEIGHTS   = 0x06   # payload: N x f32 (sync ao migrar função)
CMD_REWARD        = 0x07   # payload: u8 passou_obstaculo, u8 morreu -> f32 reward

# --- PC -> Arduino : comandos MONITOR (dump do ATmega328P) ------------------
CMD_PORTS         = 0x10   # PORTB/C/D + DDRx + PINx
CMD_READ_SRAM     = 0x11   # payload: u16 addr, u8 len
CMD_READ_EEPROM   = 0x12   # payload: u16 addr, u8 len
CMD_READ_FLASH    = 0x13   # payload: u16 addr, u8 len
CMD_TIMERS        = 0x14   # TCNT0/1/2 + prescalers (TCCRx)
CMD_ADC           = 0x15   # ADMUX, ADCH:ADCL
CMD_REGS          = 0x16   # SREG (flags) + registradores diversos
CMD_NET_ADDR      = 0x17   # endereço-base da rede na SRAM + tamanho

# Resposta sempre ecoa o `cmd` recebido. Comandos neurais que executam conta
# no Arduino devolvem, ANTES do payload de resultado, um campo u32 `t_compute`
# em microssegundos (medido com micros() no sketch) -> alimenta o benchmark.

# --- IDs de função delegável (espelha bench/delegator) ----------------------
FUNC_FORWARD      = 0
FUNC_BACKPROP     = 1
FUNC_UPDATE_W     = 2
FUNC_EXTRACT      = 3
FUNC_REWARD       = 4

LOC_PC      = 0
LOC_ARDUINO = 1

# --- IDs de algoritmo -------------------------------------------------------
ALGO_PG   = 0
ALGO_DQN  = 1
ALGO_NEAT = 2


def checksum(cmd: int, payload: bytes) -> int:
    """XOR de cmd ^ len ^ todos os bytes do payload."""
    c = cmd ^ (len(payload) & 0xFF)
    for b in payload:
        c ^= b
    return c & 0xFF


def encode(cmd: int, payload: bytes = b"") -> bytes:
    """Monta um frame completo pronto pra enviar pela serial."""
    if len(payload) > 255:
        raise ValueError("payload > 255 bytes não cabe no campo len (u8)")
    return bytes([START, cmd, len(payload)]) + payload + bytes([checksum(cmd, payload)])


def pack_floats(*values: float) -> bytes:
    """Empacota floats como f32 little-endian."""
    return struct.pack("<%df" % len(values), *values)


def unpack_floats(data: bytes) -> list[float]:
    """Desempacota uma sequência de f32 little-endian."""
    n = len(data) // 4
    return list(struct.unpack("<%df" % n, data[: n * 4]))


def unpack_u32(data: bytes, offset: int = 0) -> int:
    return struct.unpack_from("<I", data, offset)[0]


# TODO: implementar o decoder de frames (máquina de estados) em serial_comm.py,
#       lidando com ressincronização quando o START se perde.
