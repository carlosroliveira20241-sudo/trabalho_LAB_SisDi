"""Monitor interno do ATmega328P (atende a Atividade do PDF).

Usa os comandos CMD_* de MONITOR para ler o estado interno do AVR em tempo real:
portas, timers, ADC, flags (SREG) e dump de SRAM/FLASH/EEPROM. Também resolve o
endereço-base onde a rede neural vive na SRAM, para destacar os pesos no dump.

A ordem dos bytes de cada resposta espelha `arduino/dino_neural/monitor.h`.

Mapa de memória do ATmega328P (referência):
  SRAM   : 0x0100 .. 0x08FF (2 KB)  | registradores I/O em 0x20..0xFF
  EEPROM : 0x0000 .. 0x03FF (1 KB)
  FLASH  : 0x0000 .. 0x7FFF (32 KB)
"""
from __future__ import annotations

from dataclasses import dataclass

from . import protocol
from .serial_comm import SerialComm

SRAM_START, SRAM_END = 0x0100, 0x08FF
EEPROM_SIZE = 0x0400
FLASH_SIZE = 0x8000

FLAG_NAMES = ["I", "T", "H", "S", "V", "N", "Z", "C"]   # SREG, do bit 7 ao 0


@dataclass
class Ports:
    portb: int; ddrb: int; pinb: int
    portc: int; ddrc: int; pinc: int
    portd: int; ddrd: int; pind: int


@dataclass
class Regs:
    sreg: int; mcusr: int; spl: int; sph: int

    def flags(self) -> dict[str, int]:
        """SREG bit a bit: I T H S V N Z C (do mais alto para o mais baixo)."""
        return {n: (self.sreg >> (7 - i)) & 1 for i, n in enumerate(FLAG_NAMES)}

    @property
    def sp(self) -> int:
        return (self.sph << 8) | self.spl


@dataclass
class Timers:
    tcnt0: int; tcnt1: int; tcnt2: int
    tccr0b: int; tccr1b: int; tccr2b: int

    @staticmethod
    def prescaler(tccrb: int) -> str:
        # CS bits (b2..b0) -> divisor (timer0/1; o timer2 difere, simplificado)
        table = {0: "parado", 1: "/1", 2: "/8", 3: "/64", 4: "/256", 5: "/1024"}
        return table.get(tccrb & 0x07, "ext")


@dataclass
class Adc:
    admux: int; adcsra: int; value: int   # value: 0..1023

    @property
    def channel(self) -> int:
        return self.admux & 0x0F

    @property
    def volts(self) -> float:
        return self.value / 1023.0 * 5.0


@dataclass
class Buttons:
    jump: bool
    duck: bool


class Monitor:
    def __init__(self, serial_comm: SerialComm):
        self.serial = serial_comm
        self.net_addr: int | None = None
        self.net_size: int = 0

    def read_ports(self) -> Ports:
        p = self.serial.request(protocol.CMD_PORTS).payload
        return Ports(*p[:9])

    def read_regs(self) -> Regs:
        p = self.serial.request(protocol.CMD_REGS).payload
        return Regs(p[0], p[1], p[2], p[3])

    def read_timers(self) -> Timers:
        p = self.serial.request(protocol.CMD_TIMERS).payload
        tcnt1 = p[1] | (p[2] << 8)
        return Timers(p[0], tcnt1, p[3], p[4], p[5], p[6])

    def read_adc(self) -> Adc:
        p = self.serial.request(protocol.CMD_ADC).payload
        return Adc(p[0], p[1], p[2] | (p[3] << 8))

    def read_sram(self, addr: int, length: int) -> bytes:
        payload = addr.to_bytes(2, "little") + bytes([length])
        return self.serial.request(protocol.CMD_READ_SRAM, payload).payload

    def read_eeprom(self, addr: int, length: int) -> bytes:
        payload = addr.to_bytes(2, "little") + bytes([length])
        return self.serial.request(protocol.CMD_READ_EEPROM, payload).payload

    def read_flash(self, addr: int, length: int) -> bytes:
        payload = addr.to_bytes(2, "little") + bytes([length])
        return self.serial.request(protocol.CMD_READ_FLASH, payload).payload

    def read_buttons(self) -> Buttons:
        """Lê os dois pushbuttons ligados direto nos pinos digitais (pull-up
        interno, sem protoboard): bit0=jump, bit1=duck, 1=pressionado."""
        mask = self.serial.request(protocol.CMD_BUTTONS).payload[0]
        return Buttons(jump=bool(mask & 0x01), duck=bool(mask & 0x02))

    def resolve_net_addr(self) -> int:
        """Pergunta ao Arduino onde a rede neural está na SRAM (CMD_NET_ADDR)."""
        p = self.serial.request(protocol.CMD_NET_ADDR).payload
        self.net_addr = p[0] | (p[1] << 8)
        self.net_size = p[2] | (p[3] << 8)
        return self.net_addr
