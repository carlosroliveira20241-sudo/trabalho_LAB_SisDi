"""Diagnóstico dos dois pushbuttons (D2=jump, D3=duck) em tempo real.

Fica lendo CMD_BUTTONS em loop e imprime o estado bruto — sem passar pelo
resto do app (jogo, threads, UI). Aperte os botões e observe o terminal:
se nunca aparecer "PRESSIONADO", o problema é na fiação/continuidade
(D2/D3 -> botão -> GND), não no firmware nem no app.

Uso (de dentro de pc/):
    python diag_buttons.py            # usa COM4
    python diag_buttons.py COM3
"""
from __future__ import annotations

import sys
import time

import serial

from comm import protocol

port = sys.argv[1] if len(sys.argv) > 1 else "COM4"
baud = int(sys.argv[2]) if len(sys.argv) > 2 else 250000

print(f"--- abrindo {port} @ {baud} ---")
ser = serial.Serial(port, baud, timeout=0.5)
print("aberto. esperando 3s o reset/bootloader da placa...")
time.sleep(3.0)
ser.reset_input_buffer()

print("\nLendo D2 (jump) / D3 (duck) — aperte os botões. Ctrl+C para sair.\n")
frame = protocol.encode(protocol.CMD_BUTTONS)
last = None
try:
    while True:
        ser.reset_input_buffer()
        ser.write(frame)
        resp = ser.read(8)
        if not resp:
            print("(sem resposta — placa não respondeu a tempo)")
        else:
            # frame: [0xAA][cmd][len=1][mask][chk]
            mask = resp[3] if len(resp) >= 5 else None
            if mask is None:
                print("resposta incompleta:", resp.hex())
            else:
                jump = bool(mask & 0x01)
                duck = bool(mask & 0x02)
                state = (jump, duck)
                if state != last:
                    print(f"D2(jump)={'PRESSIONADO' if jump else 'solto'}   "
                          f"D3(duck)={'PRESSIONADO' if duck else 'solto'}")
                    last = state
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    ser.close()
