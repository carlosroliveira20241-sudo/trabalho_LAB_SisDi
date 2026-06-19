/*
 * neural_net.h — rede feed-forward na SRAM do ATmega328P (header-only).
 *
 * Topologia FIXA (igual a pc/neural/neural_net.py) para permitir sync de pesos:
 *   entrada(3) -> oculta(4) -> saída(2)   [tanh na oculta, softmax na saída]
 *
 * Pesos:  W1[3][4] + b1[4] + W2[4][2] + b2[2] = 26 floats = 104 bytes.
 * São variáveis globais (SRAM); o endereço é exportado via CMD_NET_ADDR para o
 * dump da aba Memória conseguir apontar exatamente para os pesos.
 *
 * Sem FPU: as contas usam float em software (~10x mais lento que o PC) — é
 * justamente o que o benchmark de delegação mede.
 */
#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <Arduino.h>
#include <math.h>
#include <string.h>

#define N_IN  4
#define N_HID 4
#define N_OUT 3
#define N_WEIGHTS (N_IN*N_HID + N_HID + N_HID*N_OUT + N_OUT)  // 35

// Pesos em SRAM (globais -> endereço estável p/ o dump).
float W1[N_IN][N_HID];
float b1[N_HID];
float W2[N_HID][N_OUT];
float b2[N_OUT];

// Inicializa com um padrão determinístico (o PC sincroniza os reais depois).
inline void net_init() {
  for (uint8_t i = 0; i < N_IN; i++)
    for (uint8_t j = 0; j < N_HID; j++)
      W1[i][j] = 0.1f * (float)(i - j);
  for (uint8_t j = 0; j < N_HID; j++) b1[j] = 0.0f;
  for (uint8_t j = 0; j < N_HID; j++)
    for (uint8_t k = 0; k < N_OUT; k++)
      W2[j][k] = 0.1f * (float)(j - k);
  for (uint8_t k = 0; k < N_OUT; k++) b2[k] = 0.0f;
}

// Forward pass com softmax na saída.
inline void net_forward(const float in[N_IN], float out[N_OUT]) {
  float h[N_HID];
  for (uint8_t j = 0; j < N_HID; j++) {
    float s = b1[j];
    for (uint8_t i = 0; i < N_IN; i++) s += in[i] * W1[i][j];
    h[j] = tanhf(s);
  }
  float z[N_OUT];
  for (uint8_t k = 0; k < N_OUT; k++) {
    float s = b2[k];
    for (uint8_t j = 0; j < N_HID; j++) s += h[j] * W2[j][k];
    z[k] = s;
  }
  // softmax geral sobre N_OUT saídas
  float m = z[0];
  for (uint8_t k = 1; k < N_OUT; k++) if (z[k] > m) m = z[k];
  float sum = 0.0f;
  for (uint8_t k = 0; k < N_OUT; k++) { out[k] = expf(z[k] - m); sum += out[k]; }
  for (uint8_t k = 0; k < N_OUT; k++) out[k] /= sum;
}

// Serializa/sincroniza os pesos na MESMA ordem do PC (W1, b1, W2, b2).
inline void net_get_weights(float *flat) {
  uint8_t n = 0;
  for (uint8_t i = 0; i < N_IN; i++)  for (uint8_t j = 0; j < N_HID; j++) flat[n++] = W1[i][j];
  for (uint8_t j = 0; j < N_HID; j++) flat[n++] = b1[j];
  for (uint8_t i = 0; i < N_HID; i++) for (uint8_t k = 0; k < N_OUT; k++) flat[n++] = W2[i][k];
  for (uint8_t k = 0; k < N_OUT; k++) flat[n++] = b2[k];
}

inline void net_set_weights(const float *flat) {
  uint8_t n = 0;
  for (uint8_t i = 0; i < N_IN; i++)  for (uint8_t j = 0; j < N_HID; j++) W1[i][j] = flat[n++];
  for (uint8_t j = 0; j < N_HID; j++) b1[j] = flat[n++];
  for (uint8_t i = 0; i < N_HID; i++) for (uint8_t k = 0; k < N_OUT; k++) W2[i][k] = flat[n++];
  for (uint8_t k = 0; k < N_OUT; k++) b2[k] = flat[n++];
}

inline uint16_t net_sram_addr() { return (uint16_t)(uintptr_t)(&W1[0][0]); }
inline uint16_t net_sram_size() { return (uint16_t)(sizeof(float) * N_WEIGHTS); }

#endif // NEURAL_NET_H
