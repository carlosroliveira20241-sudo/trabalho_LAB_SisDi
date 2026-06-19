/*
 * policy_gradient.h — REINFORCE rodando completo no Arduino.
 *
 * Espelha pc/neural/neural_net.py (backward/apply_gradients): o passo de
 * treino é dividido em DUAS funções delegáveis de verdade, cada uma com
 * implementação própria em PC e Arduino:
 *
 *   pg_backward(estado, ação, advantage) -> gradiente (N_WEIGHTS floats)
 *       calcula d(log prob da ação)/d(peso) * advantage, sem tocar nos pesos.
 *   pg_apply(gradiente, lr)
 *       aplica o passo de gradiente ASCENT: W += lr * grad.
 *
 * Isso é o que CMD_BACKPROP / CMD_UPDATE_W chamam no .ino.
 */
#ifndef POLICY_GRADIENT_H
#define POLICY_GRADIENT_H

#include <Arduino.h>
#include <math.h>
#include "neural_net.h"

// Forward + gradiente da log-prob da softmax (REINFORCE), na MESMA ordem de
// net_get_weights (W1, b1, W2, b2) -> grad deve ter N_WEIGHTS floats.
inline void pg_backward(const float in[N_IN], uint8_t action, float advantage,
                         float *grad) {
  float h[N_HID];
  for (uint8_t j = 0; j < N_HID; j++) {
    float s = b1[j];
    for (uint8_t i = 0; i < N_IN; i++) s += in[i] * W1[i][j];
    h[j] = tanhf(s);
  }
  float z2[N_OUT];
  for (uint8_t k = 0; k < N_OUT; k++) {
    float s = b2[k];
    for (uint8_t j = 0; j < N_HID; j++) s += h[j] * W2[j][k];
    z2[k] = s;
  }
  float m = z2[0];
  for (uint8_t k = 1; k < N_OUT; k++) if (z2[k] > m) m = z2[k];
  float probs[N_OUT], sum = 0.0f;
  for (uint8_t k = 0; k < N_OUT; k++) { probs[k] = expf(z2[k] - m); sum += probs[k]; }
  for (uint8_t k = 0; k < N_OUT; k++) probs[k] /= sum;

  float dz2[N_OUT];
  for (uint8_t k = 0; k < N_OUT; k++)
    dz2[k] = (((k == action) ? 1.0f : 0.0f) - probs[k]) * advantage;

  float da1[N_HID];
  for (uint8_t j = 0; j < N_HID; j++) {
    float s = 0.0f;
    for (uint8_t k = 0; k < N_OUT; k++) s += W2[j][k] * dz2[k];
    da1[j] = s;
  }
  float dz1[N_HID];
  for (uint8_t j = 0; j < N_HID; j++) dz1[j] = da1[j] * (1.0f - h[j] * h[j]);

  uint8_t n = 0;
  for (uint8_t i = 0; i < N_IN; i++)
    for (uint8_t j = 0; j < N_HID; j++) grad[n++] = in[i] * dz1[j];   // dW1
  for (uint8_t j = 0; j < N_HID; j++) grad[n++] = dz1[j];             // db1
  for (uint8_t j = 0; j < N_HID; j++)
    for (uint8_t k = 0; k < N_OUT; k++) grad[n++] = h[j] * dz2[k];    // dW2
  for (uint8_t k = 0; k < N_OUT; k++) grad[n++] = dz2[k];             // db2
}

// Aplica o gradiente (mesma ordem de net_set_weights): W += lr * grad.
inline void pg_apply(const float *grad, float lr) {
  uint8_t n = 0;
  for (uint8_t i = 0; i < N_IN; i++)
    for (uint8_t j = 0; j < N_HID; j++) W1[i][j] += lr * grad[n++];
  for (uint8_t j = 0; j < N_HID; j++) b1[j] += lr * grad[n++];
  for (uint8_t j = 0; j < N_HID; j++)
    for (uint8_t k = 0; k < N_OUT; k++) W2[j][k] += lr * grad[n++];
  for (uint8_t k = 0; k < N_OUT; k++) b2[k] += lr * grad[n++];
}

#endif // POLICY_GRADIENT_H
