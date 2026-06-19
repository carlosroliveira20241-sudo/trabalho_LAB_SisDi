/*
 * monitor.h — leitura do estado interno do ATmega328P (Atividade do PDF).
 *
 * Header-only. Cada handler preenche um buffer de saída e devolve o nº de bytes.
 * Tudo aqui é leitura direta de registradores/memória do AVR — barato e seguro.
 *
 * Mapa de memória (ATmega328P):
 *   SRAM   0x0100..0x08FF (2 KB)   — registradores I/O em 0x20..0xFF
 *   EEPROM 0x0000..0x03FF (1 KB)   — via <avr/eeprom.h>
 *   FLASH  0x0000..0x7FFF (32 KB)  — via pgm_read_byte (<avr/pgmspace.h>)
 */
#ifndef MONITOR_H
#define MONITOR_H

#include <Arduino.h>
#include <avr/io.h>
#include <avr/pgmspace.h>
#include <avr/eeprom.h>

// PORTB/C/D + DDRx + PINx -> 9 bytes (ordem: PORT, DDR, PIN por grupo).
inline uint8_t mon_ports(uint8_t *o) {
  o[0] = PORTB; o[1] = DDRB; o[2] = PINB;
  o[3] = PORTC; o[4] = DDRC; o[5] = PINC;
  o[6] = PORTD; o[7] = DDRD; o[8] = PIND;
  return 9;
}

// SREG (flags) + alguns registradores de status -> 4 bytes.
inline uint8_t mon_regs(uint8_t *o) {
  o[0] = SREG;     // flags: I T H S V N Z C
  o[1] = MCUSR;    // causa do último reset
  o[2] = SPL;      // stack pointer (low/high)
  o[3] = SPH;
  return 4;
}

// TCNT0/1/2 + prescalers (TCCRxB) -> 7 bytes.
inline uint8_t mon_timers(uint8_t *o) {
  o[0] = TCNT0;
  uint16_t t1 = TCNT1;          // 16 bits
  o[1] = t1 & 0xFF; o[2] = t1 >> 8;
  o[3] = TCNT2;
  o[4] = TCCR0B; o[5] = TCCR1B; o[6] = TCCR2B;
  return 7;
}

// ADMUX + ADCSRA + resultado (10 bits) de uma leitura do canal A0 -> 4 bytes.
inline uint8_t mon_adc(uint8_t *o) {
  int v = analogRead(A0);
  o[0] = ADMUX; o[1] = ADCSRA;
  o[2] = v & 0xFF; o[3] = (v >> 8) & 0xFF;
  return 4;
}

inline uint8_t mon_read_sram(uint16_t addr, uint8_t len, uint8_t *o) {
  volatile uint8_t *p = (volatile uint8_t *)addr;
  for (uint8_t i = 0; i < len; i++) o[i] = p[i];
  return len;
}

inline uint8_t mon_read_eeprom(uint16_t addr, uint8_t len, uint8_t *o) {
  for (uint8_t i = 0; i < len; i++) o[i] = eeprom_read_byte((const uint8_t *)(addr + i));
  return len;
}

inline uint8_t mon_read_flash(uint16_t addr, uint8_t len, uint8_t *o) {
  for (uint8_t i = 0; i < len; i++) o[i] = pgm_read_byte((const uint8_t *)(addr + i));
  return len;
}

#endif // MONITOR_H
