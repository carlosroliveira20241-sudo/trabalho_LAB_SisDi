/*
 * protocol.h — framing e comandos do protocolo serial (lado Arduino).
 *
 * Espelho de pc/comm/protocol.py. Framing:
 *   [0xAA] [cmd] [len] [payload...(len)] [checksum]
 *   checksum = XOR(cmd ^ len ^ payload[*])
 *
 * Floats: IEEE-754 32 bits little-endian (layout nativo do AVR-GCC).
 * Header-only: as funções são `inline` e definidas aqui (incluídas só pelo .ino).
 */
#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <Arduino.h>
#include <string.h>

#define START 0xAA

#define CMD_PING         0x00   // ecoa o payload de volta (teste de conexão)

// --- PC -> Arduino : NEURAIS ---
#define CMD_FORWARD      0x01
#define CMD_BACKPROP     0x02
#define CMD_DELEGATE     0x03
#define CMD_GET_WEIGHTS  0x04
#define CMD_SET_ALGO     0x05
#define CMD_SET_WEIGHTS  0x06
#define CMD_REWARD       0x07

// --- PC -> Arduino : MONITOR (dump) ---
#define CMD_PORTS        0x10
#define CMD_READ_SRAM    0x11
#define CMD_READ_EEPROM  0x12
#define CMD_READ_FLASH   0x13
#define CMD_TIMERS       0x14
#define CMD_ADC          0x15
#define CMD_REGS         0x16
#define CMD_NET_ADDR     0x17

// IDs de função delegável
#define FUNC_FORWARD   0
#define FUNC_BACKPROP  1
#define FUNC_UPDATE_W  2
#define FUNC_EXTRACT   3
#define FUNC_REWARD    4

#define LOC_PC      0
#define LOC_ARDUINO 1

#define MAX_PAYLOAD 160   // >= 35 floats (140 bytes) p/ sincronizar os pesos

struct Frame {
  uint8_t cmd;
  uint8_t len;
  uint8_t payload[MAX_PAYLOAD];
};

// Lê um frame completo da Serial. Retorna true se um frame válido chegou.
// Não bloqueia se não há nada chegando; usa o timeout da Serial p/ o resto.
inline bool readFrame(Frame *f) {
  if (Serial.available() < 1) return false;
  if (Serial.read() != START) return false;          // ressincroniza no START

  uint8_t hdr[2];
  if (Serial.readBytes(hdr, 2) < 2) return false;    // cmd, len
  f->cmd = hdr[0];
  f->len = hdr[1];
  if (f->len > MAX_PAYLOAD) return false;

  if (f->len && Serial.readBytes(f->payload, f->len) < f->len) return false;

  uint8_t chk;
  if (Serial.readBytes(&chk, 1) < 1) return false;

  uint8_t c = f->cmd ^ f->len;
  for (uint8_t i = 0; i < f->len; i++) c ^= f->payload[i];
  return chk == c;                                   // descarta se checksum ruim
}

// Envia um frame de resposta (cmd ecoa o comando recebido).
inline void sendFrame(uint8_t cmd, const uint8_t *payload, uint8_t len) {
  uint8_t c = cmd ^ len;
  Serial.write(START);
  Serial.write(cmd);
  Serial.write(len);
  for (uint8_t i = 0; i < len; i++) { Serial.write(payload[i]); c ^= payload[i]; }
  Serial.write(c);
}

// Resposta de comando neural: u32 t_compute (micros) + resultado.
inline void sendNeuralResult(uint8_t cmd, uint32_t t_compute_us,
                             const uint8_t *result, uint8_t result_len) {
  uint8_t buf[MAX_PAYLOAD];
  memcpy(buf, &t_compute_us, 4);                     // little-endian (AVR)
  if (result_len) memcpy(buf + 4, result, result_len);
  sendFrame(cmd, buf, 4 + result_len);
}

#endif // PROTOCOL_H
