"""Monitor interno do ATmega328P — dashboard ao vivo no terminal (Atividade PDF).

Mostra em tempo real, lendo o estado interno do AVR pela serial:
  - Portas digitais PORTB/C/D (com DDR e PIN), bit a bit;
  - Flags do SREG (I T H S V N Z C) + stack pointer / MCUSR;
  - Timers TCNT0/1/2 + prescalers;
  - ADC (canal A0): valor, tensão e registradores;
  - Dump de SRAM/EEPROM/FLASH, com a REGIÃO DA REDE NEURAL na SRAM destacada.

Uso (de dentro de pc/):
    python dump_monitor.py                 # escolhe a porta no terminal
    python dump_monitor.py --port COM4
"""
from __future__ import annotations

import argparse
import time

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel

from comm import protocol
from comm.serial_comm import SerialComm
from comm.port_select import choose_port
from comm.monitor import Monitor, Timers


def bits(v: int) -> str:
    """8 bits do MSB ao LSB: 1 = verde, 0 = apagado (ASCII, p/ qualquer terminal)."""
    return " ".join(
        "[bold green]1[/]" if (v >> (7 - i)) & 1 else "[dim]0[/]" for i in range(8))


def bar(value: int, vmax: int, width: int = 24) -> str:
    n = int(value / vmax * width) if vmax else 0
    return "[cyan]" + "#" * n + "[/]" + "[dim]" + "-" * (width - n) + "[/]"


def hexdump(data: bytes, base: int, highlight: range | None = None) -> str:
    lines = []
    for off in range(0, len(data), 16):
        chunk = data[off:off + 16]
        cells = []
        for i, b in enumerate(chunk):
            h = f"{b:02X}"
            if highlight and (base + off + i) in highlight:
                h = f"[reverse yellow]{h}[/]"
            cells.append(h)
        lines.append(f"[cyan]{base + off:04X}[/]  " + " ".join(cells))
    return "\n".join(lines) if lines else "(vazio)"


def ports_panel(p) -> Panel:
    def row(name, port, ddr, pin):
        return (f"[bold]PORT{name}[/]\n"
                f"  PORT {bits(port)}  0x{port:02X}\n"
                f"  DDR  {bits(ddr)}  0x{ddr:02X}\n"
                f"  PIN  {bits(pin)}  0x{pin:02X}")
    body = "\n".join([
        row("B", p.portb, p.ddrb, p.pinb),
        row("C", p.portc, p.ddrc, p.pinc),
        row("D", p.portd, p.ddrd, p.pind),
    ])
    return Panel(body, title="Portas digitais  (bit7..bit0)", border_style="blue")


def flags_panel(r) -> Panel:
    fl = r.flags()
    line = "   ".join(
        f"[bold]{n}[/] [{'green' if v else 'red dim'}]{v}[/]" for n, v in fl.items())
    extra = f"\n\nSREG 0x{r.sreg:02X}    SP 0x{r.sp:04X}    MCUSR 0x{r.mcusr:02X}"
    return Panel(line + extra, title="Flags do SREG", border_style="magenta")


def timers_panel(t) -> Panel:
    body = (f"TCNT0 [yellow]{t.tcnt0:3d}[/]   8-bit   presc {Timers.prescaler(t.tccr0b)}\n"
            f"TCNT1 [yellow]{t.tcnt1:5d}[/] 16-bit   presc {Timers.prescaler(t.tccr1b)}\n"
            f"TCNT2 [yellow]{t.tcnt2:3d}[/]   8-bit   presc {Timers.prescaler(t.tccr2b)}")
    return Panel(body, title="Temporizadores", border_style="green")


def adc_panel(a) -> Panel:
    body = (f"canal  A{a.channel}\n"
            f"valor  [yellow]{a.value:4d}[/] / 1023    [yellow]{a.volts:.2f} V[/]\n"
            f"{bar(a.value, 1023)}\n"
            f"ADMUX  {bits(a.admux)}\nADCSRA {bits(a.adcsra)}")
    return Panel(body, title="ADC (conversor A/D)", border_style="cyan")


def build(mon: Monitor) -> Group:
    ports = mon.read_ports()
    regs = mon.read_regs()
    timers = mon.read_timers()
    adc = mon.read_adc()

    sram = mon.read_sram(mon.net_addr, 32)
    net_region = range(mon.net_addr, mon.net_addr + mon.net_size)
    eeprom = mon.read_eeprom(0x0000, 16)
    flash = mon.read_flash(0x0000, 16)

    top = Columns([ports_panel(ports),
                   Group(flags_panel(regs), timers_panel(timers), adc_panel(adc))])

    mem = Panel(
        f"[bold]SRAM[/] @ 0x{mon.net_addr:04X}  "
        f"([reverse yellow] amarelo [/] = pesos da rede, {mon.net_size} bytes)\n"
        + hexdump(sram, mon.net_addr, net_region)
        + f"\n\n[bold]EEPROM[/] @ 0x0000\n" + hexdump(eeprom, 0x0000)
        + f"\n\n[bold]FLASH[/] @ 0x0000\n" + hexdump(flash, 0x0000),
        title="Dump de memória (tempo real)", border_style="yellow")

    return Group(top, mem)


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitor interno do ATmega328P")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=250000)
    args = ap.parse_args()

    port = choose_port(args.port)
    if not port:
        print("Nenhuma porta selecionada.")
        return

    console = Console()
    print(f"Conectando em {port}...")
    sc = SerialComm(port, args.baud)
    sc.open()
    mon = Monitor(sc)
    try:
        if sc.request(protocol.CMD_PING, b"\x01").payload != b"\x01":
            print("PING falhou — o sketch certo está na placa?")
            return
        mon.resolve_net_addr()
        console.print(f"rede neural na SRAM em 0x{mon.net_addr:04X} "
                      f"({mon.net_size} bytes). Ctrl+C para sair.\n")
        with Live(build(mon), console=console, refresh_per_second=4, screen=True) as live:
            while True:
                live.update(build(mon))
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        sc.close()


if __name__ == "__main__":
    main()
