/*
 * dino_neural.ino — sketch do Arduino Uno (ATmega328P).
 *
 * Passo atual: BENCHMARK DE DELEGAÇÃO. O Arduino executa o forward pass de uma
 * rede fixa (3->4->2) e devolve, junto do resultado, o tempo de cálculo medido
 * com micros() — é isso que alimenta o benchmark PC vs Arduino.
 *
 * Também responde aos comandos de MONITOR/dump (portas, timers, ADC, flags,
 * SRAM/FLASH/EEPROM) — a atividade do monitor interno do ATmega328P.
 *
 * Loop: lê um frame da serial, despacha pelo comando e responde.
 * Baud 250000 para reduzir o overhead de transporte.
 */
#include "protocol.h"
#include "neural_net.h"
#include "monitor.h"

void setup() {
  Serial.begin(250000);
  Serial.setTimeout(100);      // timeout p/ readBytes (ms)
  net_init();
}

void loop() {
  Frame f;
  if (!readFrame(&f)) return;  // nenhum frame válido neste ciclo

  switch (f.cmd) {

    case CMD_PING:               // ecoa o payload de volta (teste de conexão)
      sendFrame(CMD_PING, f.payload, f.len);
      break;

    case CMD_FORWARD: {          // forward pass cronometrado
      float in[N_IN], out[N_OUT];
      memcpy(in, f.payload, sizeof(in));
      uint32_t t0 = micros();
      net_forward(in, out);
      uint32_t dt = micros() - t0;
      sendNeuralResult(CMD_FORWARD, dt, (uint8_t *)out, sizeof(out));
      break;
    }

    case CMD_GET_WEIGHTS: {      // PC lê os pesos do Arduino (p/ comparar saídas)
      float w[N_WEIGHTS];
      net_get_weights(w);
      sendFrame(CMD_GET_WEIGHTS, (uint8_t *)w, sizeof(w));
      break;
    }

    case CMD_SET_WEIGHTS:        // PC sincroniza os pesos no Arduino
      net_set_weights((const float *)f.payload);
      sendFrame(CMD_SET_WEIGHTS, NULL, 0);   // ACK
      break;

    case CMD_REWARD: {           // calcula_recompensa delegada (cronometrada)
      uint8_t passed = f.payload[0], died = f.payload[1];
      uint32_t t0 = micros();
      float r = 1.0f + (passed ? 10.0f : 0.0f) + (died ? -100.0f : 0.0f);
      uint32_t dt = micros() - t0;
      sendNeuralResult(CMD_REWARD, dt, (uint8_t *)&r, sizeof(r));
      break;
    }

    case CMD_DELEGATE:           // confirma mudança de localização (ACK)
      sendFrame(CMD_DELEGATE, NULL, 0);
      break;

    // ---------- MONITOR / dump do ATmega328P ----------
    case CMD_PORTS:  { uint8_t b[9]; sendFrame(CMD_PORTS,  b, mon_ports(b));  break; }
    case CMD_REGS:   { uint8_t b[4]; sendFrame(CMD_REGS,   b, mon_regs(b));   break; }
    case CMD_TIMERS: { uint8_t b[7]; sendFrame(CMD_TIMERS, b, mon_timers(b)); break; }
    case CMD_ADC:    { uint8_t b[4]; sendFrame(CMD_ADC,    b, mon_adc(b));    break; }

    case CMD_READ_SRAM: {
      uint16_t a = f.payload[0] | (f.payload[1] << 8);
      uint8_t n = f.payload[2]; if (n > 64) n = 64;
      uint8_t b[64];
      sendFrame(CMD_READ_SRAM, b, mon_read_sram(a, n, b));
      break;
    }
    case CMD_READ_EEPROM: {
      uint16_t a = f.payload[0] | (f.payload[1] << 8);
      uint8_t n = f.payload[2]; if (n > 64) n = 64;
      uint8_t b[64];
      sendFrame(CMD_READ_EEPROM, b, mon_read_eeprom(a, n, b));
      break;
    }
    case CMD_READ_FLASH: {
      uint16_t a = f.payload[0] | (f.payload[1] << 8);
      uint8_t n = f.payload[2]; if (n > 64) n = 64;
      uint8_t b[64];
      sendFrame(CMD_READ_FLASH, b, mon_read_flash(a, n, b));
      break;
    }
    case CMD_NET_ADDR: {         // onde a rede vive na SRAM (p/ ver os pesos)
      uint16_t a = net_sram_addr(), s = net_sram_size();
      uint8_t b[4] = { (uint8_t)a, (uint8_t)(a >> 8), (uint8_t)s, (uint8_t)(s >> 8) };
      sendFrame(CMD_NET_ADDR, b, 4);
      break;
    }

    default:
      break;                     // comando desconhecido: ignora
  }
}
