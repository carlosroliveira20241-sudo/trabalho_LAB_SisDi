"""Rede neural feed-forward (numpy) — a implementação de PC do forward/backprop.

Topologia FIXA, igual à do Arduino (arduino/neural_net.h), para que os pesos
possam ser sincronizados byte-a-byte ao migrar funções entre os lados:

    entrada(4) -> oculta(4) -> saída(3)      [tanh na oculta, softmax na saída]

Entradas:  [dist, vel, topo_obstaculo, base_obstaculo]
Saídas:    [nada, pular, abaixar]
Contagem de pesos:
    W1: 4x4 = 16   b1: 4
    W2: 4x3 = 12   b2: 3
    total = 35 floats  (= 140 bytes na sincronização)
"""
from __future__ import annotations

import numpy as np

N_IN, N_HID, N_OUT = 4, 4, 3


def dino_policy_weights() -> np.ndarray:
    """Pesos calibrados à mão para a rede 4->4->3 jogar cactos E voadores.

    Os quatro neurônios ocultos viram detectores a partir das entradas:
      h0 = "obstáculo perto"   (limiar em dist < ~0.125)
      h1 = "obstáculo aéreo"   (limiar em topo_obstaculo < ~0.55)
      h2 = reflete a velocidade do jogo   (acende, sem alterar a decisão)
      h3 = reflete a base do obstáculo    (acende, sem alterar a decisão)
    A camada de saída combina h0 e h1:
      PULAR   = perto E de chão  (perto e NÃO aéreo)
      ABAIXAR = perto E aéreo
      NADA    = caso contrário (baseline)
    Score ~104 em cactos+voadores. Usado para a rede que migra PC<->Arduino de
    fato controlar o dino na interface.
    """
    W1 = np.zeros((N_IN, N_HID))
    W1[0, 0] = -30.0    # dist -> h0 (perto)
    W1[2, 1] = -20.0    # topo -> h1 (aéreo)
    W1[1, 2] = 3.0      # vel  -> h2 (só p/ acender)
    W1[3, 3] = 4.0      # base -> h3 (só p/ acender)
    b1 = np.array([3.75, 11.0, 0.0, -2.5])
    W2 = np.zeros((N_HID, N_OUT))
    W2[0, 1] = 3.0; W2[1, 1] = -3.0     # h0->PULAR(+), h1->PULAR(-)
    W2[0, 2] = 3.0; W2[1, 2] = 3.0      # h0->ABAIXAR(+), h1->ABAIXAR(+)
    b2 = np.array([3.0, 0.0, 0.0])      # NADA = baseline
    return np.concatenate([W1.ravel(), b1, W2.ravel(), b2]).astype(np.float32)


class NeuralNet:
    def __init__(self, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (N_IN, N_HID))
        self.b1 = np.zeros(N_HID)
        self.W2 = rng.normal(0, 0.5, (N_HID, N_OUT))
        self.b2 = np.zeros(N_OUT)
        self._cache = {}

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        z1 = x @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        out = self._softmax(z2)
        self._cache = {"x": x, "a1": a1, "z2": z2, "out": out}
        return out

    @staticmethod
    def _softmax(z):
        e = np.exp(z - np.max(z))
        return e / e.sum()

    def backward(self, x, action: int, advantage: float) -> np.ndarray:
        """Gradiente da log-prob da softmax (REINFORCE), sem tocar nos pesos.

        Espelha pg_backward no Arduino byte-a-byte: mesma fórmula, mesma ordem
        de saída (dW1, db1, dW2, db2) que `get_weights`/`set_weights`, para que
        o gradiente calculado num lado seja aplicável no outro.
        """
        x = np.asarray(x, dtype=np.float32)
        z1 = x @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = self._softmax(z2)

        dz2 = -probs
        dz2[action] += 1.0          # onehot - probs
        dz2 *= advantage            # direção de subida escalada pelo retorno

        dW2 = np.outer(a1, dz2)
        db2 = dz2
        da1 = self.W2 @ dz2
        dz1 = da1 * (1.0 - a1 * a1)  # derivada do tanh
        dW1 = np.outer(x, dz1)
        db1 = dz1

        return np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2]).astype(np.float32)

    def apply_gradients(self, grad, lr: float) -> None:
        """Aplica o passo de gradiente ASCENT: W += lr * grad (mesma ordem de get_weights)."""
        grad = np.asarray(grad, dtype=np.float32)
        i = 0
        def take(n):
            nonlocal i
            chunk = grad[i:i + n]; i += n
            return chunk
        self.W1 += lr * take(N_IN * N_HID).reshape(N_IN, N_HID)
        self.b1 += lr * take(N_HID)
        self.W2 += lr * take(N_HID * N_OUT).reshape(N_HID, N_OUT)
        self.b2 += lr * take(N_OUT)

    # --- (de)serialização para sync com o Arduino ---------------------------
    def get_weights(self) -> np.ndarray:
        """Vetor 1D com todos os pesos, na MESMA ordem do Arduino."""
        return np.concatenate([
            self.W1.ravel(), self.b1, self.W2.ravel(), self.b2
        ]).astype(np.float32)

    def set_weights(self, flat) -> None:
        flat = np.asarray(flat, dtype=np.float32)
        i = 0
        def take(n):
            nonlocal i
            chunk = flat[i:i + n]; i += n
            return chunk
        self.W1 = take(N_IN * N_HID).reshape(N_IN, N_HID)
        self.b1 = take(N_HID)
        self.W2 = take(N_HID * N_OUT).reshape(N_HID, N_OUT)
        self.b2 = take(N_OUT)
