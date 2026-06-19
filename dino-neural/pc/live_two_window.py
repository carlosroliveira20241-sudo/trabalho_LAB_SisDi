"""live_two_window.py — Treino por imitação (botões) -> Arduino jogando sozinho.

Fluxo de duas fases (substitui o autoaprendizado por RL deste ponto de entrada):

  FASE 1 — TREINO (você joga): dois pushbuttons na breadboard, ligados direto
    em pinos digitais do Arduino com pull-up interno (perna no GND, sem
    resistor). O PC lê os botões via CMD_BUTTONS, aplica sua ação no jogo e
    GRAVA cada par (estado, sua ação) — a rede ainda não muda.

  FASE 2 — ARDUINO JOGA SOZINHO: ao clicar "Enviar p/ Arduino e jogar", a rede
    treina (imitação: advantage=1.0 fixo) com TUDO que foi gravado na fase 1,
    os pesos resultantes são enviados ao Arduino (CMD_SET_WEIGHTS) e
    `forward_pass` passa a ser delegado a ele — o Arduino decide a ação a
    cada frame; o PC só aplica no jogo e mede o tempo (benchmark).

Duas janelas SDL simultâneas (pygame._sdl2.video, multi-window):
  - "Dino Neural — Jogo": o dino jogando, fase atual, estado dos botões.
  - "Dino Neural — Memória": dump ao vivo de SRAM/EEPROM/FLASH do Arduino,
    com a região dos pesos da rede destacada — visível o tempo todo enquanto
    o dino pula na outra janela.

Uso (de dentro de pc/):
    python live_two_window.py                # escolhe a porta no terminal
    python live_two_window.py --port COM4
"""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import pygame
from pygame._sdl2.video import Window, Renderer, Texture

from comm import protocol
from comm.serial_comm import SerialComm
from comm.port_select import choose_port
from comm.delegator import Router
from comm.monitor import Monitor, SRAM_START, SRAM_END, EEPROM_SIZE, FLASH_SIZE
from bench.benchmark import Benchmark
from neural.neural_net import NeuralNet, dino_policy_weights
from game.dino_game_headless import DinoGameHeadless, ACTION_JUMP, ACTION_DUCK, ACTION_NONE
from game.obstacle import CACTUS_AND_DUCK, Kind
from game import physics

PHASE_TRAIN = "train"        # você joga, o PC grava (sem treinar ainda)
PHASE_TRAINING = "training"  # treino em background (thread própria) — jogo pausado
PHASE_AUTO = "auto"          # Arduino (ou PC, em fallback) jogando sozinho

GAME_W, GAME_H = 760, 460
MEM_W, MEM_H = 720, 860

# limites válidos de cada região de memória (ver comm/monitor.py)
MEM_BOUNDS = {
    "SRAM": (SRAM_START, SRAM_END + 1),
    "EEPROM": (0, EEPROM_SIZE),
    "FLASH": (0, FLASH_SIZE),
}

BG = (18, 19, 26)
PANEL = (28, 30, 40)
PANEL2 = (36, 39, 52)
TXT = (225, 228, 235)
DIM = (120, 125, 140)
ACCENT = (90, 200, 255)
GREEN = (120, 230, 150)
RED = (230, 90, 90)


def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def heat_color(v: int):
    """0..255 -> azul (baixo) -> verde -> vermelho (alto)."""
    t = v / 255.0
    if t < 0.5:
        return lerp((30, 50, 120), (40, 200, 130), t * 2)
    return lerp((40, 200, 130), (240, 70, 60), (t - 0.5) * 2)


class LiveApp:
    def __init__(self, serial_comm):
        self.serial = serial_comm
        self.bench = Benchmark()
        self.net = NeuralNet()
        self.net.set_weights(dino_policy_weights())
        self.router = Router(serial_comm, self.bench)
        self.router.register_pc("forward_pass", lambda s: self.net.forward(s))
        self.router.register_pc("backpropagation", lambda s, a, adv: self.net.backward(s, a, adv))
        self.router.register_pc("atualiza_pesos", lambda g, lr: self.net.apply_gradients(g, lr))
        for name in self.router.location:
            self.router.location[name] = protocol.LOC_PC
        self.monitor = Monitor(serial_comm) if serial_comm else None
        if self.monitor:
            try:
                self.monitor.resolve_net_addr()
            except Exception:
                self.monitor.net_addr, self.monitor.net_size = SRAM_START, 0

        self.game = DinoGameHeadless(kinds=CACTUS_AND_DUCK)
        self.phase = PHASE_TRAIN
        self.demo_states: list = []
        self.demo_actions: list = []
        self.last_buttons = (False, False)
        self.status_msg = "Jogue com os botões — o dino imita você depois."
        self.training_busy = False

        self.mem_region = "SRAM"
        self.mem_cache = b""
        self.mem_stale = False
        self.mem_base = self.monitor.net_addr if self.monitor else SRAM_START

        self.game_buttons: dict[str, pygame.Rect] = {}
        self.mem_buttons: dict[str, pygame.Rect] = {}

        # estado mantido fresco por _io_loop (thread própria) — o render nunca
        # bloqueia esperando a serial: lê/escreve só essas variáveis simples.
        self._btn_state = (False, False)     # último estado dos botões (Fase 1)
        self._pending_state: list | None = None  # estado aguardando decisão remota (Fase 3)
        self._auto_action = ACTION_NONE          # última decisão remota conhecida
        self._stop_io = threading.Event()
        self._io_thread = threading.Thread(target=self._io_loop, daemon=True)
        self._io_thread.start()

    # ---------------- I/O serial em thread própria (não bloqueia o render) --
    def _io_loop(self):
        """Só esta thread fala com a serial fora do treino: leitura dos botões
        (Fase 1), decisão remota do forward_pass (Fase 3) e dump de memória —
        exatamente a separação que o README descreve ("jogo desacoplado do
        serial"), que `step_train`/`step_auto`/`refresh_memory` violavam ao
        chamar a serial direto no loop de render."""
        next_mem = 0.0
        while not self._stop_io.is_set():
            did_work = False
            if self.phase == PHASE_TRAIN and self.monitor is not None:
                try:
                    b = self.monitor.read_buttons()
                    self._btn_state = (b.jump, b.duck)
                    did_work = True
                except Exception:
                    pass
            elif (self.phase == PHASE_AUTO
                    and self.router.location["forward_pass"] == protocol.LOC_ARDUINO
                    and self._pending_state is not None):
                state, self._pending_state = self._pending_state, None
                try:
                    probs = self.router.call("forward_pass", state)
                    self._auto_action = int(np.argmax(probs))
                except Exception as e:
                    self.status_msg = f"forward_pass falhou ({e}) — voltando pro PC"
                    self.router.location["forward_pass"] = protocol.LOC_PC
                did_work = True

            now = time.time()
            if self.monitor is not None and now >= next_mem:
                next_mem = now + 0.3
                self._refresh_memory_once()
                did_work = True

            if not did_work:
                time.sleep(0.01)

    # ---------------- leitura dos pushbuttons ----------------
    def _read_buttons(self):
        if self.monitor is not None:
            return self._btn_state            # mantido fresco por _io_loop
        keys = pygame.key.get_pressed()  # fallback p/ testar sem a placa
        return bool(keys[pygame.K_UP] or keys[pygame.K_SPACE]), bool(keys[pygame.K_DOWN])

    # ---------------- fase 1: você joga, o PC grava ----------------
    def step_train(self):
        jump, duck = self._read_buttons()
        self.last_buttons = (jump, duck)
        action = ACTION_JUMP if jump else (ACTION_DUCK if duck else ACTION_NONE)
        state = self.game.features()
        self.game.step(action)
        self.demo_states.append(state)
        self.demo_actions.append(action)
        if self.game.is_dead:
            self.game.reset()

    # ---------------- fase 2: Arduino decide, PC só aplica ----------------
    def step_auto(self):
        state = self.game.features()
        if self.router.location["forward_pass"] == protocol.LOC_ARDUINO:
            # decisão remota é assíncrona (via _io_loop): aplica a última
            # conhecida agora e entrega o estado atual pra próxima rodada —
            # ~1 frame de atraso, mas o render nunca espera o round-trip.
            self._pending_state = state
            action = self._auto_action
        else:
            probs = self.router.call("forward_pass", state)  # local: instantâneo
            action = int(np.argmax(probs))
        self.game.step(action)
        if self.game.is_dead:
            self.game.reset()

    def step(self):
        if self.phase == PHASE_TRAIN:
            self.step_train()
        elif self.phase == PHASE_AUTO:
            self.step_auto()
        # PHASE_TRAINING: jogo pausado enquanto o treino roda em background

    # ---------------- transição treino -> Arduino ----------------
    def send_to_arduino(self):
        """Dispara o treino numa thread própria — a UI (as duas janelas)
        continua respondendo enquanto o treino (que pode envolver milhares de
        frames gravados) roda em background."""
        if self.phase != PHASE_TRAIN or self.training_busy:
            return
        n = len(self.demo_states)
        if n == 0:
            self.status_msg = "Nada gravado ainda — jogue um pouco primeiro."
            return
        states, actions = list(self.demo_states), list(self.demo_actions)
        self.demo_states.clear()
        self.demo_actions.clear()
        self.training_busy = True
        self.phase = PHASE_TRAINING
        self.status_msg = f"Treinando com {n} jogadas..."
        threading.Thread(target=self._train_worker, args=(states, actions), daemon=True).start()

    @staticmethod
    def _balance_classes(states, actions):
        """A maioria dos frames gravados é ACTION_NONE (você só aperta botão
        perto do obstáculo) — sem isso, o gradiente médio fica dominado por
        'não fazer nada'. Mantém todos os frames de pular/abaixar e amostra só
        uma quantidade equivalente de frames de 'nada'."""
        actions_arr = np.asarray(actions)
        none_idx = np.flatnonzero(actions_arr == ACTION_NONE)
        other_idx = np.flatnonzero(actions_arr != ACTION_NONE)
        if len(other_idx) == 0 or len(none_idx) <= len(other_idx):
            return states, actions
        rng = np.random.default_rng()
        keep_none = rng.choice(none_idx, size=len(other_idx), replace=False)
        keep = np.sort(np.concatenate([keep_none, other_idx]))
        return [states[i] for i in keep], [int(actions_arr[i]) for i in keep]

    def _train_worker(self, states, actions):
        """Roda fora da thread de render: calcula o gradiente médio (sobre o
        conjunto balanceado) e aplica UMA única atualização — mais estável que
        milhares de passos sequenciais de lr fixo — depois envia os pesos ao
        Arduino e migra forward_pass pra ele via Router.move (mesma convenção
        de delegação/CMD_DELEGATE usada no resto do projeto)."""
        try:
            states_b, actions_b = self._balance_classes(states, actions)
            grads = [self.router.call("backpropagation", s, a, 1.0) for s, a in zip(states_b, actions_b)]
            mean_grad = np.mean(grads, axis=0)
            self.router.call("atualiza_pesos", mean_grad, 0.3)
            if self.serial:
                self.router.move("forward_pass", protocol.LOC_ARDUINO, sync_state=self._sync_weights_move)
                self.status_msg = (f"Treinado com {len(states)} jogadas "
                                    f"({len(states_b)} balanceadas) — Arduino jogando sozinho.")
            else:
                self.status_msg = (f"Treinado com {len(states)} jogadas "
                                    f"({len(states_b)} balanceadas) — sem Arduino, IA fica no PC.")
        except Exception as e:
            self.status_msg = f"Falha no treino/envio dos pesos: {e}"
        finally:
            self.game.reset()
            self.phase = PHASE_AUTO
            self.training_busy = False

    def _sync_weights_move(self, frm, to):
        """`sync_state` do Router.move: PC->Arduino manda os pesos treinados;
        Arduino->PC lê de volta (espelha o padrão de studio.py)."""
        if to == protocol.LOC_ARDUINO:
            self.serial.request(protocol.CMD_SET_WEIGHTS,
                                 protocol.pack_floats(*self.net.get_weights()))
        else:
            resp = self.serial.request(protocol.CMD_GET_WEIGHTS)
            self.net.set_weights(protocol.unpack_floats(resp.payload))

    def back_to_training(self):
        if self.phase != PHASE_AUTO:
            return
        if self.serial and self.router.location["forward_pass"] == protocol.LOC_ARDUINO:
            try:
                self.router.move("forward_pass", protocol.LOC_PC, sync_state=self._sync_weights_move)
            except Exception as e:
                self.status_msg = f"Erro ao voltar pro PC: {e}"
                self.router.location["forward_pass"] = protocol.LOC_PC
        self.phase = PHASE_TRAIN
        self.status_msg = "De volta ao treino — jogue para gravar de novo."
        self.game.reset()

    # ---------------- memória (dump ao vivo) ----------------
    def _refresh_memory_once(self):
        """Lê 256 bytes da região atual. Chamado só por _io_loop (serial é
        bloqueante) — nunca direto do loop de render."""
        reader = {"SRAM": self.monitor.read_sram,
                  "EEPROM": self.monitor.read_eeprom,
                  "FLASH": self.monitor.read_flash}[self.mem_region]
        base = self.mem_base
        try:
            data = bytearray()
            for off in range(0, 256, 64):       # 256 bytes em 4 leituras
                data += reader(base + off, 64)
            self.mem_cache = bytes(data)
            self.mem_stale = False
        except Exception:
            self.mem_stale = True               # leitura falhou: mantém o cache, mas avisa

    # ---------------- cliques ----------------
    def click_game(self, pos):
        for sid, r in self.game_buttons.items():
            if r.collidepoint(pos):
                if sid == "transition":
                    if self.phase == PHASE_TRAIN:
                        self.send_to_arduino()
                    elif self.phase == PHASE_AUTO:
                        self.back_to_training()
                    # PHASE_TRAINING: botão ocupado, clique ignorado
                break

    def click_memory(self, pos):
        for sid, r in self.mem_buttons.items():
            if not r.collidepoint(pos):
                continue
            if sid.startswith("mem_"):
                self.mem_region = sid[4:]
                self.mem_base = (self.monitor.net_addr if (self.mem_region == "SRAM" and self.monitor)
                                  else MEM_BOUNDS[self.mem_region][0])
                self.mem_cache = b""
                self._mem_timer = 0.0
            elif sid == "page_prev":
                self._page_memory(-256)
            elif sid == "page_next":
                self._page_memory(256)
            break

    def _page_memory(self, delta):
        """Navega em blocos de 256 bytes dentro da região, sem sair dos
        limites válidos (a UI antes só mostrava sempre os primeiros 256
        bytes de EEPROM/FLASH, sem jeito de ver o resto)."""
        lo, hi = MEM_BOUNDS[self.mem_region]
        self.mem_base = max(lo, min(hi - 256, self.mem_base + delta))
        self.mem_cache = b""
        self._mem_timer = 0.0

    # ---------------- desenho: janela do JOGO ----------------
    def draw_game(self, font, big, small) -> pygame.Surface:
        surf = pygame.Surface((GAME_W, GAME_H))
        surf.fill(BG)
        on = self.serial is not None
        col = GREEN if on else RED
        txt = "Arduino: conectado" if on else "Arduino: offline (sem botões/placa)"
        pygame.draw.circle(surf, col, (20, 18), 6)
        surf.blit(small.render(txt, True, DIM), (34, 11))

        if self.phase == PHASE_TRAIN:
            phase_lbl, phase_col = "FASE 1 — VOCÊ JOGA (botões)", ACCENT
        elif self.phase == PHASE_TRAINING:
            phase_lbl, phase_col = "TREINANDO — aguarde...", (255, 200, 60)
        else:
            on_arduino = self.router.location["forward_pass"] == protocol.LOC_ARDUINO
            phase_lbl = "FASE 2 — ARDUINO JOGA SOZINHO" if on_arduino else "FASE 2 — PC JOGA SOZINHO (fallback)"
            phase_col = (255, 165, 60) if on_arduino else (200, 160, 90)
        surf.blit(big.render(phase_lbl, True, phase_col), (16, 34))

        # estado dos botões (feedback visual de que a placa está respondendo)
        jump, duck = self.last_buttons
        bx = GAME_W - 220
        for i, (lbl, pressed) in enumerate((("PULAR", jump), ("ABAIXAR", duck))):
            r = pygame.Rect(bx + i * 110, 16, 96, 32)
            pygame.draw.rect(surf, GREEN if pressed else PANEL2, r, border_radius=8)
            pygame.draw.rect(surf, GREEN if pressed else (60, 64, 80), r, 1, border_radius=8)
            t = small.render(lbl, True, BG if pressed else DIM)
            surf.blit(t, (r.x + (r.width - t.get_width()) // 2, r.y + 9))

        if self.phase == PHASE_TRAIN:
            surf.blit(small.render(f"gravado: {len(self.demo_states)} jogadas", True, DIM), (16, 66))
        elif self.phase == PHASE_TRAINING:
            surf.blit(small.render("calculando gradiente e sincronizando com o Arduino...", True, DIM), (16, 66))
        else:
            loc = "ARDUINO" if self.router.location["forward_pass"] == protocol.LOC_ARDUINO else "PC"
            surf.blit(small.render(f"forward_pass rodando em: {loc}", True, DIM), (16, 66))

        # botão de transição
        if self.phase == PHASE_TRAIN:
            lbl, bcol = "Enviar p/ Arduino e jogar", (180, 140, 255)
        elif self.phase == PHASE_TRAINING:
            lbl, bcol = "Treinando...", (140, 140, 150)
        else:
            lbl, bcol = "Voltar a treinar", (230, 90, 90)
        tr = pygame.Rect(16, GAME_H - 96, 280, 36)
        pygame.draw.rect(surf, PANEL2, tr, border_radius=10)
        pygame.draw.rect(surf, bcol, tr, 1, border_radius=10)
        t = font.render(lbl, True, bcol)
        surf.blit(t, (tr.x + (tr.width - t.get_width()) // 2, tr.y + 9))
        self.game_buttons = {"transition": tr}

        surf.blit(small.render(self.status_msg, True, DIM), (16, GAME_H - 52))

        # cena do jogo
        game_rect = pygame.Rect(16, 96, GAME_W - 32, GAME_H - 96 - 44)
        gs = pygame.Surface(game_rect.size)
        gs.fill((247, 247, 247))
        ground_y = int(physics.GROUND_Y * game_rect.height / physics.HEIGHT)
        pygame.draw.line(gs, (83, 83, 83), (0, ground_y), (game_rect.width, ground_y), 2)
        sx, sy = game_rect.width / physics.WIDTH, game_rect.height / physics.HEIGHT
        def scaled(hitbox):
            x, y, w, h = hitbox
            return [int(x * sx), int(y * sy), int(w * sx), int(h * sy)]
        pygame.draw.rect(gs, (83, 83, 83), scaled(self.game.dino.hitbox))
        for o in self.game.obstacles:
            color = (210, 120, 40) if o.kind == Kind.PTERODACTYL_LOW else (60, 120, 60)
            pygame.draw.rect(gs, color, scaled(o.hitbox))
        gs.blit(small.render(f"score {self.game.score}", True, (83, 83, 83)),
                (game_rect.width - 90, 10))
        surf.blit(gs, game_rect.topleft)
        return surf

    # ---------------- desenho: janela de MEMÓRIA ----------------
    def draw_memory(self, font, big, small) -> pygame.Surface:
        surf = pygame.Surface((MEM_W, MEM_H))
        surf.fill(BG)
        surf.blit(big.render("Dump ao vivo do ATmega328P", True, TXT), (20, 18))

        if not self.monitor:
            surf.blit(font.render("Conecte o Arduino para ver a memória.", True, DIM), (20, 60))
            return surf

        bx = 20
        self.mem_buttons = {}
        for reg in ("SRAM", "EEPROM", "FLASH"):
            r = pygame.Rect(bx, 56, 96, 30)
            on = reg == self.mem_region
            pygame.draw.rect(surf, PANEL2 if on else PANEL, r, border_radius=8)
            pygame.draw.rect(surf, ACCENT if on else (60, 64, 80), r, 1, border_radius=8)
            surf.blit(small.render(reg, True, TXT if on else DIM), (r.x + 16, r.y + 7))
            self.mem_buttons["mem_" + reg] = r
            bx += 106

        # paginação dentro da região (antes só dava pra ver os primeiros 256 bytes)
        lo, hi = MEM_BOUNDS[self.mem_region]
        prev_r = pygame.Rect(bx + 14, 56, 36, 30)
        next_r = pygame.Rect(bx + 56, 56, 36, 30)
        pygame.draw.rect(surf, PANEL2, prev_r, border_radius=8)
        pygame.draw.rect(surf, PANEL2, next_r, border_radius=8)
        surf.blit(small.render("<", True, TXT), (prev_r.x + 13, prev_r.y + 7))
        surf.blit(small.render(">", True, TXT), (next_r.x + 13, next_r.y + 7))
        self.mem_buttons["page_prev"] = prev_r
        self.mem_buttons["page_next"] = next_r
        # linha própria (não ao lado dos botões): com região longa o texto
        # passava da borda direita da janela e ficava cortado.
        surf.blit(small.render(f"{self.mem_base:#06x}..{self.mem_base + 255:#06x}  (regiao: {lo:#06x}-{hi - 1:#06x})",
                                True, DIM), (20, 94))

        if self.mem_stale:
            surf.blit(small.render("dados desatualizados — sem resposta do Arduino", True, RED), (20, 112))

        data = self.mem_cache
        if not data:
            surf.blit(small.render("lendo...", True, DIM), (20, 100))
            return surf

        cols, cell = 16, 36
        gx, gy = 70, 130
        net = (self.monitor.net_addr, self.monitor.net_size)
        for idx, b in enumerate(data):
            row, c = divmod(idx, cols)
            x, y = gx + c * cell, gy + row * cell
            pygame.draw.rect(surf, heat_color(b), (x, y, cell - 4, cell - 4), border_radius=4)
            addr = self.mem_base + idx
            if self.mem_region == "SRAM" and net[0] <= addr < net[0] + net[1]:
                pygame.draw.rect(surf, (255, 220, 60), (x, y, cell - 4, cell - 4), 2, border_radius=4)
            if c == 0:
                surf.blit(small.render(f"{addr:04X}", True, DIM), (gx - 56, y + 9))

        ly = gy + -(-len(data) // cols) * cell + 18   # divisão de cols arredondada p/ cima
        surf.blit(small.render("baixo", True, DIM), (gx, ly + 3))
        for k in range(130):
            pygame.draw.rect(surf, heat_color(int(k / 130 * 255)), (gx + 50 + k * 2, ly, 2, 16))
        surf.blit(small.render("alto", True, DIM), (gx + 50 + 264, ly + 3))
        if self.mem_region == "SRAM":
            pygame.draw.rect(surf, (255, 220, 60), (gx + 4, ly + 30, 14, 14), 2)
            surf.blit(small.render("= pesos da rede neural na SRAM", True, DIM), (gx + 26, ly + 30))
        return surf


def main() -> None:
    ap = argparse.ArgumentParser(description="Dino Neural — treino por imitação + Arduino autônomo")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=250000)
    ap.add_argument("--no-serial", action="store_true")
    args = ap.parse_args()

    sc = None
    if not args.no_serial:
        port = choose_port(args.port)
        if port:
            print(f"Conectando em {port}...")
            try:
                sc = SerialComm(port, args.baud)
                sc.open()
                if sc.request(protocol.CMD_PING, b"\x01").payload != b"\x01":
                    raise RuntimeError("PING sem eco esperado (firmware errado na placa?)")
            except Exception as e:
                print(f"Não consegui conectar ({e}) — seguindo em modo offline (sem Arduino).")
                if sc is not None:
                    try:
                        sc.close()
                    except Exception:
                        pass
                sc = None
        else:
            print("Sem porta — rodando offline (sem Arduino).")

    pygame.init()
    font = pygame.font.SysFont("consolas", 16)
    big = pygame.font.SysFont("consolas", 20, bold=True)
    small = pygame.font.SysFont("consolas", 13)
    clock = pygame.time.Clock()

    win_game = Window("Dino Neural — Jogo", size=(GAME_W, GAME_H))
    win_mem = Window("Dino Neural — Memória", size=(MEM_W, MEM_H))
    r_game = Renderer(win_game)
    r_mem = Renderer(win_mem)

    app = LiveApp(sc)
    running = True
    while running:
        clock.tick(60)
        for e in pygame.event.get():
            ewin = getattr(e, "window", None)
            if e.type == pygame.WINDOWCLOSE:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if ewin is not None and ewin.id == win_game.id:
                    app.click_game(e.pos)
                elif ewin is not None and ewin.id == win_mem.id:
                    app.click_memory(e.pos)
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False

        app.step()

        game_surf = app.draw_game(font, big, small)
        tex = Texture.from_surface(r_game, game_surf)
        r_game.clear()
        tex.draw()
        r_game.present()

        mem_surf = app.draw_memory(font, big, small)
        tex2 = Texture.from_surface(r_mem, mem_surf)
        r_mem.clear()
        tex2.draw()
        r_mem.present()

    # dá um tempo pro treino em background (se ainda estiver rodando) terminar
    # antes de fechar a serial por baixo dele
    for _ in range(50):
        if not app.training_busy:
            break
        time.sleep(0.05)
    app._stop_io.set()
    app._io_thread.join(timeout=1.0)
    if sc:
        sc.close()
    pygame.quit()


if __name__ == "__main__":
    main()
