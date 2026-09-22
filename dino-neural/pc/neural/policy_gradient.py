"""REINFORCE (Policy Gradient) — algoritmo recomendado, roda completo no Arduino.

Coleta uma trajetória por episódio (estados, ações, recompensas), calcula os
retornos descontados e atualiza a política na direção que aumenta a
probabilidade das ações que levaram a retornos altos.

No PC usa a NeuralNet (numpy). Quando `forward_pass`/`backpropagation` estão
delegados, o Router envia as chamadas ao Arduino — este algoritmo não muda,
só o destino das chamadas.
"""
from __future__ import annotations

import numpy as np

GAMMA = 0.99


class PolicyGradient:
    def __init__(self, router, net, lr: float = 0.01):
        self.router = router
        self.net = net
        self.lr = lr
        self._batch = []          # episódios acumulados até o próximo update()
        self._reset_episode()

    def _reset_episode(self):
        self.states, self.actions, self.rewards = [], [], []

    def act(self, state_vec) -> int:
        """Forward (via Router -> PC ou Arduino) e amostra a ação."""
        probs = self.router.call("forward_pass", state_vec)
        action = int(np.random.choice(len(probs), p=np.asarray(probs)))
        self.states.append(state_vec)
        self.actions.append(action)
        return action

    def observe(self, reward: float) -> None:
        self.rewards.append(reward)

    def end_episode(self) -> None:
        """Fecha o episódio: guarda a trajetória no lote (não atualiza ainda)."""
        if self.rewards:
            returns = self._discounted_returns()
            self._batch.append((list(self.states), list(self.actions), returns))
        self._reset_episode()

    def update(self) -> None:
        """Atualiza os pesos com TODOS os episódios do lote de uma vez.

        Treinar em lote reduz muito a variância do gradiente — é o que permite a
        política aprender a condicionar a ação pelo estado (em vez de achatar
        numa probabilidade fixa). A vantagem é normalizada sobre o lote inteiro,
        que serve de baseline.

        Cada passo passa pelo Router em DUAS chamadas delegáveis de verdade:
        "backpropagation" (calcula o gradiente) e "atualiza_pesos" (aplica),
        cada uma podendo estar no PC ou no Arduino independentemente.
        """
        if not self._batch:
            return
        states, actions, rets = [], [], []
        for st, ac, rt in self._batch:
            states += st
            actions += ac
            rets += list(rt)
        rets = np.asarray(rets, dtype=np.float32)
        adv = (rets - rets.mean()) / (rets.std() + 1e-6)   # baseline = média do lote
        for state, action, a in zip(states, actions, adv):
            grad = self.router.call("backpropagation", state, action, float(a))
            self.router.call("atualiza_pesos", grad, self.lr)
        self._batch = []

    def _discounted_returns(self):
        g, out = 0.0, []
        for r in reversed(self.rewards):
            g = r + GAMMA * g
            out.append(g)
        out.reverse()
        return np.asarray(out, dtype=np.float32)
