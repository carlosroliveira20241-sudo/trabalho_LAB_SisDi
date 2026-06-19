"""Diagnóstico da conexão serial com o Arduino.

Mostra, em cru (hex), o que a placa devolve — ajuda a distinguir:
  - "(nada)" em tudo  -> sketch não subiu / placa não está rodando nosso código
  - bytes embaralhados -> baud errado
  - eco correto do PING -> comunicação OK (o problema era timing)

Uso (de dentro de pc/):
    python diag_serial.py            # usa COM4
    python diag_serial.py COM3       # outra porta
    python diag_serial.py COM4 9600  # testa outro baud
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

# 1. escuta bytes espontâneos (alguns sketches imprimem no boot)
print("\n[1] escutando 2s por bytes espontâneos...")
t = time.time()
boot = b""
while time.time() - t < 2.0:
    boot += ser.read(64)
print("    boot:", boot.hex() if boot else "(nada)")

# 2. envia PING algumas vezes e mostra a resposta crua
frame = protocol.encode(protocol.CMD_PING, bytes([0x01, 0x02, 0x03, 0x04]))
print(f"\n[2] enviando PING ({frame.hex()}) 5x:")
got_any = False
for i in range(5):
    ser.reset_input_buffer()
    ser.write(frame)
    time.sleep(0.3)
    resp = ser.read(64)
    if resp:
        got_any = True
    print(f"    resposta {i}: {resp.hex() if resp else '(nada)'}")

ser.close()

print("\n--- conclusão ---")
if not boot and not got_any:
    print("Nada chegou. Provável: o sketch NÃO está rodando na placa.")
    print(" -> reabra o Arduino IDE, confirme Board=Arduino Uno e Port=" + port + ",")
    print("    clique Upload e espere 'Done uploading'. Feche o Serial Monitor.")
elif got_any:
    print("A placa respondeu! Se o eco do PING apareceu (aa00...), a comunicação")
    print("está OK e o bench_arduino.py deve funcionar agora.")
else:
    print("Vieram bytes no boot mas o PING não ecoou — pode ser baud diferente.")
    print(" -> tente: python diag_serial.py " + port + " 9600")
