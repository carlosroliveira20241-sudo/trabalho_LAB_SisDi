# Dino Neural — Rede Neural Distribuída PC/Arduino com Benchmark de Delegação

Sistema distribuído onde uma rede neural aprende a jogar o jogo do dinossauro
(recriado em Pygame). As funções de cálculo/aprendizado podem ser **delegadas
dinamicamente** entre PC e Arduino Uno (ATmega328P) via USB Serial — e o tempo de
execução de cada lado é medido e exibido **em tempo real**, formando um benchmark
de offloading vivo enquanto a IA roda.

Junto disso, o app funciona como um **monitor interno do ATmega328P**: portas,
timers, ADC, flags e dump em tempo real de SRAM/FLASH/EEPROM — incluindo a região
da SRAM onde a rede neural vive (dá pra ver os pesos mudando durante o treino).

## Decisões de arquitetura

- **Hardware:** Arduino Uno (ATmega328P) fixo. 2 KB SRAM, 16 MHz, sem FPU.
  - Só **Policy Gradient** roda completo no Arduino.
  - DQN e NEAT rodam no PC; o Arduino no máximo faz o forward pass quando delegado.
- **App único** (Python) com abas (Textual). Modo terminal alternativo via `rich` (`--cli`).
- **Centro do projeto:** delegação dinâmica + benchmark em tempo real (PC vs Arduino).
- **Jogo desacoplado do serial:** o loop do Pygame roda livre; a inferência/treino roda
  em thread própria, no ritmo que o lado ativo permitir. Quando algo está no Arduino,
  o `calls/s` cai — e isso aparece no benchmark de propósito.

## Modelo de medição (núcleo do benchmark)

| Métrica         | O que é                         | Origem                                  |
|-----------------|---------------------------------|-----------------------------------------|
| `t_compute`     | só a conta em si                | PC: `perf_counter()` · Arduino: `micros()` enviado de volta |
| `t_roundtrip`   | tempo que o PC esperou (Arduino)| wall-clock send→recv (tx + compute + rx) |
| `t_serial`      | overhead de transporte          | `t_roundtrip − t_compute_arduino`        |

## Estrutura

```
dino-neural/
├── pc/
│   ├── main.py            # bootstrap: serial + threads + UI
│   ├── app.py             # app Textual com as abas
│   ├── cli.py             # modo terminal (rich Live)
│   ├── game/              # jogo do dino (pygame) + headless p/ NEAT
│   ├── neural/            # rede + algoritmos (PG, DQN, NEAT) + recompensa
│   ├── comm/              # serial, protocolo, router de delegação, monitor do AVR
│   ├── bench/             # coletor de estatísticas do benchmark
│   └── ui/                # uma aba por tela
└── arduino/
    ├── dino_neural.ino    # loop principal: parser + dispatch
    ├── neural_net.h       # forward + backprop (rede em SRAM, endereço exportado)
    ├── policy_gradient.h  # REINFORCE no Arduino
    ├── monitor.h          # leitura de SRAM/FLASH/EEPROM/regs/timers/ADC
    └── protocol.h         # framing + comandos
```

## Setup (PC)

```bash
cd pc
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py            # interface com abas (Textual)
python main.py --cli      # modo terminal (rich)
```

## Status

Esqueleto/scaffold. Cada arquivo tem assinaturas e TODOs. Preencher pilar por pilar.
