"""Aprendizado por imitação (behavioral cloning) — o dino aprende jogando você.

Reusa exatamente as mesmas funções delegáveis do REINFORCE
("backpropagation" + "atualiza_pesos" via Router), mas em vez de um retorno
descontado usa a própria ação do humano como rótulo correto e `advantage=1.0`
fixo: o gradiente só empurra a rede a aumentar a probabilidade de repetir o
que você fez naquele estado. Isso também faz a imitação exercitar o mesmo
caminho de delegação PC<->Arduino que o resto do projeto mede no benchmark.
"""
from __future__ import annotations


class ImitationLearner:
    def __init__(self, router, net, lr: float = 0.01, batch_size: int = 16):
        self.router = router
        self.net = net
        self.lr = lr
        self.batch_size = batch_size
        self._buffer = []   # [(state, action), ...] desde o último update()

    def record(self, state, action: int) -> None:
        """Guarda um par (estado, ação) observado do jogador humano."""
        self._buffer.append((state, action))
        if len(self._buffer) >= self.batch_size:
            self.update()

    def update(self) -> None:
        """Treina com todos os pares acumulados (advantage fixo = 1.0)."""
        for state, action in self._buffer:
            grad = self.router.call("backpropagation", state, action, 1.0)
            self.router.call("atualiza_pesos", grad, self.lr)
        self._buffer = []
