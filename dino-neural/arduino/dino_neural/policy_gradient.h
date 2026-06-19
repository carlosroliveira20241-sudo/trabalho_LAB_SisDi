/*
 * policy_gradient.h — REINFORCE rodando completo no Arduino.
 *
 * Atualiza os pesos da rede (neural_net.h) na direção que aumenta a
 * probabilidade das ações que renderam retornos altos. Sem replay buffer
 * (cabe nos 2 KB de SRAM), por isso é o algoritmo recomendado para o Uno.
 *
 * O PC envia CMD_BACKPROP com (estado, recompensa); aqui acumulamos a
 * trajetória ou atualizamos online, conforme a estratégia escolhida.
 */
#ifndef POLICY_GRADIENT_H
#define POLICY_GRADIENT_H

#include <Arduino.h>
#include "neural_net.h"

#define PG_LR    0.01f
#define PG_GAMMA 0.99f

// Atualiza os pesos a partir de um passo (estado, ação amostrada, retorno).
// TODO: gradiente de log-prob da softmax * retorno; passo em W1/b1/W2/b2.
void pg_update(const float in[N_IN], uint8_t action, float ret);

// Reinicia acumuladores de trajetória (chamar no fim do episódio).
void pg_reset();

#endif // POLICY_GRADIENT_H
