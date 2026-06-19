"""Benchmark de delegação: forward pass PC vs Arduino.

Fecha o ciclo do passo 2. O script:
  1. seleciona a porta do Arduino pelo terminal (ou --port);
  2. testa a conexão (PING);
  3. lê os pesos do Arduino e carrega na rede do PC (assim os dois calculam
     IGUAL -> dá pra validar que o resultado bate);
  4. roda N forward passes nos dois lados e compara os tempos.

Uso (de dentro de pc/):
    python bench_arduino.py                 # escolhe a porta no terminal
    python bench_arduino.py --port COM3      # porta fixa
    python bench_arduino.py --n 300          # nº de amostras
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np

from comm import protocol
from comm.serial_comm import SerialComm
from comm.port_select import choose_port
from neural.neural_net import NeuralNet


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark forward PC vs Arduino")
    ap.add_argument("--port", default=None, help="porta serial (ex.: COM3)")
    ap.add_argument("--baud", type=int, default=250000)
    ap.add_argument("--n", type=int, default=200, help="nº de forward passes")
    args = ap.parse_args()

    port = choose_port(args.port)
    if not port:
        print("Nenhuma porta selecionada. Saindo.")
        return

    print(f"\nConectando em {port} @ {args.baud}...")
    sc = SerialComm(port, args.baud)
    sc.open()
    try:
        # 1. PING -------------------------------------------------------------
        token = bytes([0x01, 0x02, 0x03, 0x04])
        r = sc.request(protocol.CMD_PING, token)
        if r.payload != token:
            print("PING falhou (resposta inesperada). O sketch certo está na placa?")
            return
        print(f"PING OK  (round-trip {r.t_roundtrip * 1e3:.2f} ms)")

        # 2. sincroniza pesos: lê do Arduino -> carrega no PC -----------------
        rw = sc.request(protocol.CMD_GET_WEIGHTS)
        weights = protocol.unpack_floats(rw.payload)
        net = NeuralNet()
        net.set_weights(weights)
        print(f"pesos sincronizados ({len(weights)} floats)\n")

        # 3. benchmark --------------------------------------------------------
        pc_t, ard_compute, ard_rt = [], [], []
        max_err = 0.0
        for _ in range(args.n):
            state = [random.uniform(0, 1) for _ in range(4)]

            t0 = time.perf_counter()
            out_pc = net.forward(state)
            pc_t.append(time.perf_counter() - t0)

            resp = sc.request(protocol.CMD_FORWARD, protocol.pack_floats(*state))
            t_compute = protocol.unpack_u32(resp.payload, 0) / 1e6
            out_ard = protocol.unpack_floats(resp.payload[4:])
            ard_compute.append(t_compute)
            ard_rt.append(resp.t_roundtrip)
            max_err = max(max_err, abs(out_pc[0] - out_ard[0]))

        # 4. resultados -------------------------------------------------------
        def ms(x):
            return f"{np.mean(x) * 1e3:8.3f} ms"

        def us(x):
            return f"{np.mean(x) * 1e6:8.1f} us"

        pc_mean = np.mean(pc_t)
        ard_c_mean = np.mean(ard_compute)
        serial_overhead = np.mean(ard_rt) - ard_c_mean

        print("=" * 52)
        print(f"  amostras: {args.n}      (resultado PC vs Arduino batem: "
              f"erro máx {max_err:.2e})")
        print("-" * 52)
        print(f"  PC   forward (compute) : {us(pc_t)}")
        print(f"  ARD  forward (compute) : {ms(ard_compute)}")
        print(f"  ARD  serial (transporte): {serial_overhead * 1e3:8.3f} ms")
        print(f"  ARD  round-trip total  : {ms(ard_rt)}")
        print("-" * 52)
        print(f"  Arduino é ~{ard_c_mean / pc_mean:.0f}x mais lento no cálculo")
        print(f"  e ~{np.mean(ard_rt) / pc_mean:.0f}x considerando o serial")
        print("=" * 52)

    finally:
        sc.close()


if __name__ == "__main__":
    main()
