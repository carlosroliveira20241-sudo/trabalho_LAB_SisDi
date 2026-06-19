"""Seleção da porta serial do Arduino pelo terminal.

- `available_ports()` -> lista as portas disponíveis (com descrição).
- `auto_detect()`  -> tenta achar o Arduino pela descrição/fabricante.
- `choose_port()`  -> menu interativo no terminal; aceita também um --port fixo.

Uso direto (de dentro de pc/):
    python -m comm.port_select        # só lista e deixa escolher
"""
from __future__ import annotations

from serial.tools import list_ports as _list_ports

# pistas comuns no nome/descrição de placas Arduino (Uno, clones CH340, etc.)
# inclui variantes em português que o Windows mostra ("Dispositivo Serial USB").
HINTS = ("arduino", "ch340", "ch341", "usb-serial", "usb serial", "serial usb",
         "dispositivo serial", "usb serial device", "wch", "uno", "2341")


def available_ports() -> list:
    """Portas seriais disponíveis, ordenadas pelo nome (COM3, COM4...)."""
    return sorted(_list_ports.comports(), key=lambda p: p.device)


def auto_detect() -> str | None:
    """Devolve a primeira porta que parece ser um Arduino, ou None."""
    for p in available_ports():
        blob = f"{p.description} {p.manufacturer or ''} {p.hwid}".lower()
        if any(h in blob for h in HINTS):
            return p.device
    return None


def print_ports() -> list:
    ports = available_ports()
    if not ports:
        print("Nenhuma porta serial encontrada. O Arduino está conectado?")
        return ports
    print("Portas disponíveis:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device:<8} {p.description}")
    return ports


def choose_port(preferred: str | None = None) -> str | None:
    """Resolve a porta a usar.

    1. Se `preferred` foi passado (--port), usa ela.
    2. Senão, tenta auto-detectar o Arduino.
    3. Senão, mostra um menu no terminal pra você escolher.
    Devolve o nome da porta (ex.: "COM3") ou None se cancelar/não houver.
    """
    if preferred:
        return preferred

    ports = print_ports()
    if not ports:
        return None

    auto = auto_detect()
    if auto:
        print(f"\nArduino detectado em {auto}.")
        resp = input(f"Usar {auto}? [Enter=sim / nº de outra porta / q=sair]: ").strip()
        if resp == "":
            return auto
        if resp.lower() == "q":
            return None
        # cai para a seleção por número abaixo
    else:
        resp = input("\nDigite o número da porta [q=sair]: ").strip()
        if resp.lower() == "q":
            return None

    try:
        return ports[int(resp)].device
    except (ValueError, IndexError):
        print("Seleção inválida.")
        return None


if __name__ == "__main__":
    port = choose_port()
    print(f"\nPorta escolhida: {port}" if port else "\nNenhuma porta escolhida.")
