"""Dino Neural Studio — interface gráfica única (pygame).

Menu lateral com três telas:
  [1] PRINCIPAL  — a rede neural desenhada (neurônios acendem conforme ativados,
                   arestas coloridas pelos pesos) em cima; o jogo do dino embaixo.
  [2] MEMÓRIA    — mapa de calor da memória do Arduino (SRAM/EEPROM/FLASH), com a
                   região dos pesos da rede destacada.
  [3] DELEGAÇÃO  — troca cada função entre PC e Arduino com um clique, mostrando o
                   custo (PC vs Arduino) em barras, ao vivo, enquanto a IA joga.

O jogo roda o tempo todo (em qualquer tela), então a delegação e o benchmark
continuam ao vivo. Use o mouse no menu ou as teclas 1/2/3.

Uso (de dentro de pc/):
    python studio.py                 # escolhe a porta no terminal
    python studio.py --port COM4
    python studio.py --no-serial     # só PC (telas Memória/Arduino desabilitadas)
"""
from __future__ import annotations

import argparse
import threading

import numpy as np
import pygame

from comm import protocol
from comm.serial_comm import SerialComm
from comm.port_select import choose_port
from comm.delegator import Router
from bench.benchmark import Benchmark
from comm.monitor import Monitor
from neural.neural_net import NeuralNet, dino_policy_weights, N_IN, N_HID, N_OUT
from neural.reward import step_reward
from neural.imitation import ImitationLearner
from game.dino_game_headless import DinoGameHeadless, ACTION_JUMP, ACTION_DUCK, ACTION_NONE
from game.obstacle import CACTUS_AND_DUCK, Kind, Obstacle
from game.dino import Dino
from game import physics

LIVE_MAX_FRAMES = 1400      # teto de uma geração no modo aprendizado ao vivo
SPAWN_GAP = 350


class PopWorld:
    """Um mundo compartilhado (obstáculos) com N dinos — um por genoma.

    Todos os dinos jogam o MESMO cenário ao mesmo tempo; cada um morre quando
    bate. Serve para ver a geração inteira jogando e morrendo na tela.
    """
    def __init__(self, n, kinds):
        self.kinds = kinds
        self.dinos = [Dino() for _ in range(n)]
        self.alive = [True] * n
        self.score = [0] * n
        self.fit = [0.0] * n
        self.speed = physics.SPEED_START
        self.obstacles = [Obstacle.random(physics.WIDTH, kinds)]
        self.frames = 0

    def _next_obstacle(self):
        ahead = [o for o in self.obstacles if o.x + o.w >= physics.DINO_X]
        return min(ahead, key=lambda o: o.x, default=None)

    def features(self):
        o = self._next_obstacle()
        if o is None:
            return [1.0, self.speed / physics.SPEED_MAX, 1.0, 1.0]
        return [(o.x - physics.DINO_X) / physics.WIDTH, self.speed / physics.SPEED_MAX,
                o.y / physics.HEIGHT, (o.y + o.h) / physics.HEIGHT]

    def n_alive(self):
        return sum(self.alive)

    def step(self, decide_fn):
        """Avança um frame. `decide_fn(i, features)` devolve a ação do dino i."""
        self.speed = min(physics.SPEED_MAX, self.speed + physics.SPEED_ACCEL)
        for o in self.obstacles:
            o.update(self.speed)
            if not o.passed and o.x + o.w < physics.DINO_X:
                o.passed = True
                for i in range(len(self.dinos)):
                    if self.alive[i]:
                        self.score[i] += 1
        self.obstacles = [o for o in self.obstacles if not o.offscreen]
        last = max(self.obstacles, key=lambda o: o.x, default=None)
        if last is None or physics.WIDTH - last.x >= SPAWN_GAP:
            self.obstacles.append(Obstacle.random(physics.WIDTH + 50, self.kinds))

        feats = self.features()
        for i, dino in enumerate(self.dinos):
            if not self.alive[i]:
                continue
            a = decide_fn(i, feats)
            dino.duck(a == ACTION_DUCK)
            if a == ACTION_JUMP:
                dino.jump()
            dino.update()
            if any(physics.aabb_collision(*dino.hitbox, *o.hitbox) for o in self.obstacles):
                self.alive[i] = False
                self.fit[i] = self.frames + 10 * self.score[i]
        self.frames += 1

    def finalize_fitness(self):
        """Fitness final (vivos no fim do tempo recebem o frame atual)."""
        for i in range(len(self.dinos)):
            if self.alive[i]:
                self.fit[i] = self.frames + 10 * self.score[i]
        return self.fit

# ---- tema ----
W, H = 1140, 720
SIDEBAR = 200
BG = (18, 19, 26)
PANEL = (28, 30, 40)
PANEL2 = (36, 39, 52)
TXT = (225, 228, 235)
DIM = (120, 125, 140)
ACCENT = (90, 200, 255)
PC_COLOR = (90, 200, 255)
ARD_COLOR = (255, 165, 60)
GREEN = (120, 230, 150)

SCREENS = [("1  Principal", "main"), ("2  Memoria", "memory"), ("3  Delegacao", "deleg")]

DELEGABLE = [
    ("forward_pass", "decide a ação (rede)", True),
    ("calcula_recompensa", "calcula o reward", True),
    ("backpropagation", "treina os pesos", True),
    ("atualiza_pesos", "aplica gradiente", True),
]


def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def heat_color(v: int):
    """0..255 -> azul (baixo) -> verde -> vermelho (alto)."""
    t = v / 255.0
    if t < 0.5:
        return lerp((30, 50, 120), (40, 200, 130), t * 2)
    return lerp((40, 200, 130), (240, 70, 60), (t - 0.5) * 2)


class Studio:
    def __init__(self, serial_comm, no_serial):
        self.no_serial = no_serial
        self.serial = serial_comm
        self.bench = Benchmark()
        self.net = NeuralNet()
        self.net.set_weights(dino_policy_weights())
        self.router = Router(serial_comm, self.bench)
        self.router.register_pc("forward_pass", lambda s: self.net.forward(s))
        self.router.register_pc("calcula_recompensa", lambda p, d: step_reward(p, d))
        self.router.register_pc("backpropagation", lambda s, a, adv: self.net.backward(s, a, adv))
        self.router.register_pc("atualiza_pesos", lambda grad, lr: self.net.apply_gradients(grad, lr))
        for name in self.router.location:
            self.router.location[name] = protocol.LOC_PC
        self.monitor = Monitor(serial_comm) if serial_comm else None
        self.imitation = ImitationLearner(self.router, self.net)

        if serial_comm:
            # sincroniza os pesos calibrados na placa e descobre onde a rede vive
            try:
                serial_comm.request(protocol.CMD_SET_WEIGHTS,
                                    protocol.pack_floats(*self.net.get_weights()))
                self.monitor.resolve_net_addr()
            except Exception:
                self.monitor.net_addr = 0x0100
                self.monitor.net_size = 0

        self.game = DinoGameHeadless(kinds=CACTUS_AND_DUCK)
        self.prev_score = 0
        self.acts = None                 # (entradas, oculta, saída) p/ a viz
        self.screen_id = "main"
        self.mem_region = "SRAM"
        self.mem_cache = b""
        self.mem_base = self.monitor.net_addr if self.monitor else 0x0100
        self._mem_timer = 0.0
        self.buttons = {}                # rects clicáveis
        # modo "jogar via botões" (imitação: o dino aprende a copiar o humano)
        self.human_mode = False
        # modo aprendizado (NEAT ao vivo, em thread separada)
        self.learning = False
        self.neat_trainer = None
        self.neat_net = None
        self.generation = 0
        self.best_fit = 0.0
        self.fit_hist = []
        self.generation_done_event = threading.Event()
        self.active_generation = False
        self.pop_world = None
        self.current_nets = []

    # ---------------- aprendizado ao vivo (NEAT) ----------------
    def toggle_learning(self):
        if self.learning:
            self.learning = False
            self.active_generation = False
            self.generation_done_event.set()  # destrava a thread de evolução
            return
        self.learning = True
        self.neat_net = None
        self.generation = 0
        self.best_fit = 0.0
        self.fit_hist = []
        self.game.reset(); self.prev_score = 0
        threading.Thread(target=self._neat_worker, daemon=True).start()

    def _neat_worker(self):
        import neat
        from neural.neat_trainer import NeatTrainer
        from game.obstacle import CACTUS_AND_DUCK
        self.neat_trainer = NeatTrainer(kinds=CACTUS_AND_DUCK)
        while self.learning:
            # Evolui usando nossa avaliação em tempo real no loop principal
            best = self.neat_trainer.pop.run(self.eval_population_realtime, 1)
            self.neat_net = neat.nn.FeedForwardNetwork.create(best, self.neat_trainer.config)
            self.generation += 1
            self.best_fit = best.fitness or 0.0
            self.fit_hist.append(self.best_fit)
            self.fit_hist = self.fit_hist[-60:]

    def eval_population_realtime(self, genomes, config) -> None:
        """Chamado pelo NEAT para avaliar a população em tempo real."""
        import neat
        self.current_nets = []
        for _gid, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            self.current_nets.append((genome, net))
            
        self.pop_world = PopWorld(len(genomes), self.game.kinds)
        self.generation_done_event.clear()
        self.active_generation = True
        
        # Espera o loop principal processar todos os dinos até a morte
        self.generation_done_event.wait()
        self.active_generation = False

    def _step_learning(self):
        if not self.active_generation or self.pop_world is None:
            return
            
        # Ação de cada dino da população
        def decide_fn(i, feats):
            genome, net = self.current_nets[i]
            # feats: [dist, vel, topo, base]
            jump_sig, duck_sig = net.activate(feats)
            if jump_sig > 0.0 and jump_sig >= duck_sig:
                return ACTION_JUMP
            if duck_sig > 0.0:
                return ACTION_DUCK
            return ACTION_NONE
            
        self.pop_world.step(decide_fn)
        
        # Encontra o líder atual (vivo com maior score, ou o primeiro vivo)
        leader_idx = -1
        max_score = -1
        for i in range(len(self.pop_world.dinos)):
            if self.pop_world.alive[i] and self.pop_world.score[i] > max_score:
                max_score = self.pop_world.score[i]
                leader_idx = i
                
        # Se todos morreram neste frame, pega o de maior score geral
        if leader_idx == -1:
            leader_idx = int(np.argmax(self.pop_world.score))
            
        genome, leader_net = self.current_nets[leader_idx]
        
        # Mapeia as ativações do líder para self.acts
        feats = self.pop_world.features()
        in_a = np.array([
            leader_net.values.get(-1, 0.0),
            leader_net.values.get(-2, 0.0),
            leader_net.values.get(-3, 0.0),
            leader_net.values.get(-4, 0.0)
        ])
        hid_a = np.array([
            leader_net.values.get(2, 0.0),
            leader_net.values.get(3, 0.0),
            leader_net.values.get(4, 0.0),
            leader_net.values.get(5, 0.0)
        ])
        jump_sig = leader_net.values.get(0, 0.0)
        duck_sig = leader_net.values.get(1, 0.0)
        out_a = np.array([
            1.0 if (jump_sig <= 0.0 and duck_sig <= 0.0) else 0.0,
            jump_sig,
            duck_sig
        ])
        self.acts = (in_a, hid_a, out_a)
        
        # Sincroniza pesos do líder com a rede de visualização rígida do Studio
        self.net.W1.fill(0.0)
        self.net.W2.fill(0.0)
        self.net.b1.fill(0.0)
        self.net.b2.fill(0.0)
        for conn_key, conn in genome.connections.items():
            if not conn.enabled:
                continue
            u, v = conn_key
            if u < 0 and v >= 2:
                idx_u = -u - 1
                idx_v = v - 2
                if 0 <= idx_u < N_IN and 0 <= idx_v < N_HID:
                    self.net.W1[idx_u, idx_v] = conn.weight
            elif u >= 2 and v == 0:
                idx_u = u - 2
                if 0 <= idx_u < N_HID:
                    self.net.W2[idx_u, 1] = conn.weight
            elif u >= 2 and v == 1:
                idx_u = u - 2
                if 0 <= idx_u < N_HID:
                    self.net.W2[idx_u, 2] = conn.weight
                    
        for node_key, node in genome.nodes.items():
            if node_key >= 2:
                idx = node_key - 2
                if 0 <= idx < N_HID:
                    self.net.b1[idx] = node.bias
            elif node_key == 0:
                self.net.b2[1] = node.bias
            elif node_key == 1:
                self.net.b2[2] = node.bias
                
        # Fim da geração (todos mortos ou frame limite atingido)
        if self.pop_world.n_alive() == 0 or self.pop_world.frames >= LIVE_MAX_FRAMES:
            fits = self.pop_world.finalize_fitness()
            for i, (gen, _) in enumerate(self.current_nets):
                gen.fitness = fits[i]
                
            best_gen = max(self.current_nets, key=lambda pair: pair[0].fitness or 0.0)[0]
            self.best_fit = best_gen.fitness or 0.0
            
            # Sincroniza pesos do melhor genoma com o Arduino (caso delegado)
            if self.serial and self.router.location["forward_pass"] == protocol.LOC_ARDUINO:
                try:
                    self._sync_weights()
                except Exception as ex:
                    print(f"[Serial] Erro ao sincronizar pesos com a placa: {ex}")
                    
            self.generation_done_event.set()

    # ---------------- jogar via botões (imitação) ----------------
    def toggle_human_mode(self):
        """Liga/desliga o modo onde VOCÊ joga (botões) e o dino imita."""
        if self.learning:
            return  # mutuamente exclusivo com o aprendizado NEAT ao vivo
        self.human_mode = not self.human_mode
        if self.human_mode:
            self.game.reset()
            self.prev_score = 0

    @staticmethod
    def _keyboard_buttons():
        """Substituto de teclado p/ testar sem a placa (sem serial -> sem CMD_BUTTONS)."""
        keys = pygame.key.get_pressed()
        return keys[pygame.K_UP] or keys[pygame.K_SPACE], keys[pygame.K_DOWN]

    def _read_buttons(self):
        if self.monitor is not None:
            try:
                btn = self.monitor.read_buttons()
                return btn.jump, btn.duck
            except Exception:
                pass
        return self._keyboard_buttons()

    def _step_human(self):
        jump, duck = self._read_buttons()
        action = ACTION_JUMP if jump else (ACTION_DUCK if duck else ACTION_NONE)
        state = self.game.features()
        self.net.forward(state)                      # popula o cache p/ a viz
        self.acts = (np.asarray(state), self.net._cache["a1"], self.net._cache["out"])
        self.game.step(action)
        self.imitation.record(state, action)          # treina a rede a imitar você
        if self.game.is_dead:
            self.game.reset()
            self.prev_score = 0

    def _pull_weights(self):
        """Lê os pesos atuais do Arduino e sincroniza no PC (Arduino -> PC)."""
        if not self.serial:
            return
        try:
            resp = self.serial.request(protocol.CMD_GET_WEIGHTS)
            self.net.set_weights(protocol.unpack_floats(resp.payload))
        except Exception:
            pass

    # ---------------- lógica (roda todo frame, em qualquer tela) ----------------
    def step_ai(self):
        if self.human_mode:
            self._step_human()
            return
        if self.learning:
            self._step_learning()
            return
        state = self.game.features()
        self.net.forward(state)                      # popula o cache p/ a viz
        self.acts = (np.asarray(state), self.net._cache["a1"], self.net._cache["out"])
        try:
            probs = self.router.call("forward_pass", state)
        except Exception as e:
            print(f"[Serial Error] forward_pass falhou: {e}. Revertendo para o PC.")
            self.router.location["forward_pass"] = protocol.LOC_PC
            probs = self.net.forward(state)
        action = int(np.argmax(probs))               # 0 nada, 1 pular, 2 abaixar
        self.game.step(action)
        passed = self.game.score > self.prev_score
        self.prev_score = self.game.score
        try:
            self.router.call("calcula_recompensa", passed, self.game.is_dead)
        except Exception:
            self.router.location["calcula_recompensa"] = protocol.LOC_PC
        if self.game.is_dead:
            self.game.reset()
            self.prev_score = 0

    def refresh_memory(self, dt):
        if not self.monitor:
            return
        self._mem_timer -= dt
        if self._mem_timer > 0:
            return
        self._mem_timer = 0.4
        reader = {"SRAM": self.monitor.read_sram,
                  "EEPROM": self.monitor.read_eeprom,
                  "FLASH": self.monitor.read_flash}[self.mem_region]
        base = self.mem_base
        try:
            data = bytearray()
            for off in range(0, 256, 64):            # 256 bytes em 4 leituras
                data += reader(base + off, 64)
            self.mem_cache = bytes(data)
        except Exception:
            pass                                     # leitura falhou: mantém o cache

    # ---------------- pesos (resetar / calibrar) ----------------
    def _sync_weights(self):
        if self.serial:
            try:
                self.serial.request(protocol.CMD_SET_WEIGHTS,
                                    protocol.pack_floats(*self.net.get_weights()))
            except Exception:
                pass

    # ---------------- desenho ----------------
    def draw(self, surf, font, big, small, clock):
        surf.fill(BG)
        self._sidebar(surf, font, big)
        area = pygame.Rect(SIDEBAR, 0, W - SIDEBAR, H)
        if self.screen_id == "main":
            self._screen_main(surf, font, big, small)
        elif self.screen_id == "memory":
            self._screen_memory(surf, font, big, small)
        else:
            self._screen_deleg(surf, font, big, small)
        # rodapé
        fps = small.render(f"{clock.get_fps():4.0f} FPS", True, DIM)
        surf.blit(fps, (W - 70, H - 22))

    def _sidebar(self, surf, font, big):
        pygame.draw.rect(surf, PANEL, (0, 0, SIDEBAR, H))
        surf.blit(big.render("Dino", True, ACCENT), (24, 26))
        surf.blit(big.render("Neural", True, TXT), (24, 54))
        self.buttons.clear()
        y = 130
        for label, sid in SCREENS:
            r = pygame.Rect(16, y, SIDEBAR - 32, 46)
            active = sid == self.screen_id
            pygame.draw.rect(surf, PANEL2 if active else PANEL, r, border_radius=10)
            if active:
                pygame.draw.rect(surf, ACCENT, (r.x, r.y, 4, r.h), border_radius=2)
            surf.blit(font.render(label, True, TXT if active else DIM), (r.x + 18, r.y + 13))
            self.buttons[sid] = r
            y += 56
        # status de conexão
        on = self.serial is not None
        col = GREEN if on else (200, 80, 80)
        txt = "Arduino: COM ON" if on else "Arduino: offline"
        pygame.draw.circle(surf, col, (28, H - 40), 6)
        surf.blit(font.render(txt, True, DIM), (42, H - 48))

    # ---- TELA PRINCIPAL: rede + jogo ----
    def _screen_main(self, surf, font, big, small):
        net_rect = pygame.Rect(SIDEBAR + 20, 20, W - SIDEBAR - 40, 360)
        game_rect = pygame.Rect(SIDEBAR + 20, 396, W - SIDEBAR - 40, 300)
        if self.human_mode:
            title = "Você está jogando — o dino aprende a imitar (botões)"
        elif self.learning:
            title = f"Aprendendo (NEAT) — Geração {self.generation}"
        else:
            title = "Rede neural  (neurônios acendem ao ativar)"
        self._panel(surf, net_rect, title)

        # botão de controle (Aprender/Parar) no canto superior direito
        learn_lbl = "Parar" if self.learning else "Aprender"
        learn_col = (230, 90, 90) if self.learning else (180, 140, 255)
        r = pygame.Rect(net_rect.right - 120, net_rect.y + 12, 100, 28)
        pygame.draw.rect(surf, PANEL2, r, border_radius=8)
        pygame.draw.rect(surf, learn_col, r, 1, border_radius=8)
        # Centraliza o texto no botão
        lbl_surf = small.render(learn_lbl, True, learn_col)
        surf.blit(lbl_surf, (r.x + (r.width - lbl_surf.get_width()) // 2, r.y + 6))
        self.buttons["net_learn"] = r

        # botão "Jogar (botões)" — modo imitação, ao lado do botão Aprender
        human_lbl = "Parar" if self.human_mode else "Jogar (botões)"
        human_col = (230, 90, 90) if self.human_mode else GREEN
        rh = pygame.Rect(r.x - 150, r.y, 140, 28)
        pygame.draw.rect(surf, PANEL2, rh, border_radius=8)
        pygame.draw.rect(surf, human_col, rh, 1, border_radius=8)
        hlbl_surf = small.render(human_lbl, True, human_col)
        surf.blit(hlbl_surf, (rh.x + (rh.width - hlbl_surf.get_width()) // 2, rh.y + 6))
        self.buttons["net_human"] = rh

        # Desenha a rede neural
        self._draw_network(surf, net_rect, font, small)

        if self.human_mode:
            playing = "você (botões) — dino imita"
        elif self.learning:
            playing = "NEAT populacional"
        else:
            playing = "rede fixa"
        self._panel(surf, game_rect, f"Jogo do dino  ({playing} jogando)")
        self._draw_game(surf, game_rect.inflate(-16, -16).move(0, 8))

    def _draw_network(self, surf, rect, font, small):
        # Estatísticas do NEAT no canto superior esquerdo
        if self.learning:
            surf.blit(font.render(f"GERAÇÃO {self.generation}", True, (180, 140, 255)), (rect.x + 20, rect.y + 46))
            surf.blit(small.render(f"Melhor fitness: {self.best_fit:.0f}", True, TXT), (rect.x + 20, rect.y + 70))
            
            # Sparkline compacto no canto esquerdo inferior
            if len(self.fit_hist) > 1:
                gx, gy, gw, gh = rect.x + 20, rect.bottom - 45, 140, 30
                lo, hi = min(self.fit_hist), max(self.fit_hist) or 1
                rng = (hi - lo) or 1
                pts = [(gx + gw * i / (len(self.fit_hist) - 1),
                        gy - gh * (v - lo) / rng) for i, v in enumerate(self.fit_hist)]
                pygame.draw.lines(surf, (180, 140, 255), False, pts, 1)
                surf.blit(small.render("Fitness histórico", True, DIM), (gx, gy + 4))

        in_lbl = ["dist", "vel", "topo", "base"]
        out_lbl = ["nada", "PULAR", "ABAIXAR"]
        x_in, x_hid, x_out = rect.x + 170, rect.centerx, rect.right - 170
        def col_y(n, i):
            top = rect.y + 70
            h = rect.height - 120
            return int(top + h * (i + 0.5) / n)
        pos_in = [(x_in, col_y(N_IN, i)) for i in range(N_IN)]
        pos_hid = [(x_hid, col_y(N_HID, i)) for i in range(N_HID)]
        pos_out = [(x_out, col_y(N_OUT, i)) for i in range(N_OUT)]

        in_a = self.acts[0] if self.acts is not None else np.zeros(N_IN)
        hid_a = self.acts[1] if self.acts is not None else np.zeros(N_HID)
        out_a = self.acts[2] if self.acts is not None else np.zeros(N_OUT)

        # arestas coloridas pelos pesos (com o valor do peso nas mais fortes)
        self._edges(surf, small, pos_in, pos_hid, self.net.W1)
        self._edges(surf, small, pos_hid, pos_out, self.net.W2)
        # neurônios: cor/brilho pela ativação + o valor numérico no nó
        self._nodes(surf, small, pos_in, in_a, (90, 200, 255), in_lbl, "left")
        self._nodes(surf, small, pos_hid, hid_a, (120, 230, 150), None, None, self.net.b1)
        self._nodes(surf, small, pos_out, out_a, (255, 165, 60), out_lbl, "right", self.net.b2)

    def _edges(self, surf, small, src, dst, Wt):
        wmax = max(1e-3, float(np.abs(Wt).max()))
        for i, a in enumerate(src):
            for j, b in enumerate(dst):
                w = float(Wt[i, j])
                inten = min(1.0, abs(w) / wmax)
                if inten < 0.04:
                    continue
                base = ARD_COLOR if w > 0 else (80, 150, 255)
                col = lerp((45, 48, 60), base, inten)
                pygame.draw.line(surf, col, a, b, 1 + int(3 * inten))
                if inten > 0.5:                       # rotula o peso das arestas fortes
                    mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
                    surf.blit(small.render(f"{w:+.1f}", True, lerp(DIM, base, 0.6)),
                              (mx - 12, my - 16))

    def _nodes(self, surf, small, positions, values, color, labels, side, bias=None):
        for i, (x, y) in enumerate(positions):
            v = float(values[i])
            a = min(1.0, abs(v))                      # intensidade do brilho
            fill = lerp((45, 48, 62), color, a)
            if a > 0.15:                              # halo quando ativo
                halo = pygame.Surface((64, 64), pygame.SRCALPHA)
                pygame.draw.circle(halo, (*color, int(90 * a)), (32, 32), int(18 + 12 * a))
                surf.blit(halo, (x - 32, y - 32))
            pygame.draw.circle(surf, fill, (x, y), 18)
            pygame.draw.circle(surf, color if a > 0.3 else (70, 74, 90), (x, y), 18, 2)
            # valor da ativação dentro do nó
            val = small.render(f"{v:+.2f}", True, (12, 14, 20) if a > 0.5 else TXT)
            surf.blit(val, (x - val.get_width() // 2, y - 7))
            if labels:                                # rótulo da entrada/saída
                t = small.render(labels[i], True, TXT)
                lx = x - 26 - t.get_width() if side == "left" else x + 26
                surf.blit(t, (lx, y - 8))
            if bias is not None:                      # viés do neurônio (seu "peso")
                bt = small.render(f"b={float(bias[i]):+.1f}", True, DIM)
                surf.blit(bt, (x - bt.get_width() // 2, y + 22))

    def _draw_game(self, surf, rect):
        gs = pygame.Surface((physics.WIDTH, physics.HEIGHT))
        gs.fill((247, 247, 247))
        pygame.draw.line(gs, (83, 83, 83), (0, physics.GROUND_Y),
                         (physics.WIDTH, physics.GROUND_Y), 2)
        
        if self.learning and self.pop_world is not None:
            # Desenha todos os dinos vivos da população do NEAT correndo em tempo real
            for i, dino in enumerate(self.pop_world.dinos):
                if self.pop_world.alive[i]:
                    # Pega a primeira ativação viva como líder para dar cor escura, outros cinza claro
                    first_alive = next((idx for idx, alive in enumerate(self.pop_world.alive) if alive), 0)
                    col = (83, 83, 83) if i == first_alive else (160, 160, 160)
                    pygame.draw.rect(gs, col, [int(v) for v in dino.hitbox])
            for o in self.pop_world.obstacles:
                col = (210, 120, 40) if o.kind == Kind.PTERODACTYL_LOW else (60, 120, 60)
                pygame.draw.rect(gs, col, [int(v) for v in o.hitbox])
            f = pygame.font.SysFont("consolas", 20)
            gs.blit(f.render(f"vivos {self.pop_world.n_alive()}/{len(self.pop_world.dinos)}", True, (83, 83, 83)), (16, 12))
            gs.blit(f.render(f"score {max(self.pop_world.score)}", True, (83, 83, 83)), (physics.WIDTH - 130, 12))
        else:
            # Desenha o dinossauro único normal
            pygame.draw.rect(gs, (83, 83, 83), [int(v) for v in self.game.dino.hitbox])
            for o in self.game.obstacles:
                col = (210, 120, 40) if o.kind == Kind.PTERODACTYL_LOW else (60, 120, 60)
                pygame.draw.rect(gs, col, [int(v) for v in o.hitbox])
            f = pygame.font.SysFont("consolas", 20)
            gs.blit(f.render(f"score {self.game.score}", True, (83, 83, 83)), (physics.WIDTH - 130, 12))
            
        scaled = pygame.transform.smoothscale(gs, (rect.width, rect.height))
        surf.blit(scaled, rect.topleft)

    # ---- TELA MEMÓRIA: mapa de calor + estatísticas/delegação ----
    def _screen_memory(self, surf, font, big, small):
        mem = pygame.Rect(SIDEBAR + 20, 20, 560, H - 40)
        stats = pygame.Rect(mem.right + 16, 20, W - mem.right - 36, H - 40)
        self._panel(surf, mem, "Mapa de calor da memória")
        self._panel(surf, stats, "Estatísticas de processamento")
        self._proc_stats(surf, font, small, stats)

        if not self.monitor:
            surf.blit(font.render("Conecte o Arduino", True, DIM), (mem.x + 30, mem.y + 70))
            surf.blit(small.render("para ler a memória.", True, DIM), (mem.x + 30, mem.y + 94))
            return
        # botões de região
        bx = mem.x + 24
        for reg in ("SRAM", "EEPROM", "FLASH"):
            r = pygame.Rect(bx, mem.y + 46, 86, 28)
            on = reg == self.mem_region
            pygame.draw.rect(surf, PANEL2 if on else PANEL, r, border_radius=8)
            pygame.draw.rect(surf, ACCENT if on else (60, 64, 80), r, 1, border_radius=8)
            surf.blit(small.render(reg, True, TXT if on else DIM), (r.x + 14, r.y + 6))
            self.buttons["mem_" + reg] = r
            bx += 96

        data = self.mem_cache
        if not data:
            return
        cols, cell = 16, 30
        gx, gy = mem.x + 50, mem.y + 100
        net = (self.monitor.net_addr, self.monitor.net_size)
        for idx, b in enumerate(data):
            row, c = divmod(idx, cols)
            x, y = gx + c * cell, gy + row * cell
            pygame.draw.rect(surf, heat_color(b), (x, y, cell - 3, cell - 3), border_radius=4)
            addr = self.mem_base + idx
            if self.mem_region == "SRAM" and net[0] <= addr < net[0] + net[1]:
                pygame.draw.rect(surf, (255, 220, 60), (x, y, cell - 3, cell - 3), 2, border_radius=4)
            if c == 0:
                surf.blit(small.render(f"{addr:04X}", True, DIM), (gx - 46, y + 7))
        # legenda
        ly = gy + (len(data) // cols + 1) * cell + 16
        surf.blit(small.render("baixo", True, DIM), (gx, ly + 3))
        for k in range(110):
            pygame.draw.rect(surf, heat_color(int(k / 110 * 255)), (gx + 46 + k * 2, ly, 2, 15))
        surf.blit(small.render("alto", True, DIM), (gx + 46 + 224, ly + 3))
        if self.mem_region == "SRAM":
            pygame.draw.rect(surf, (255, 220, 60), (gx + 4, ly + 26, 13, 13), 2)
            surf.blit(small.render("= pesos da rede na SRAM", True, DIM), (gx + 24, ly + 26))

    def _proc_stats(self, surf, font, small, rect):
        """Estatísticas de processamento + toggles de delegação (na tela Memória)."""
        y = rect.y + 56
        for name, _desc, ready in DELEGABLE:
            if not ready:
                continue
            loc = self.router.location.get(name, protocol.LOC_PC)
            on_ard = loc == protocol.LOC_ARDUINO
            col = ARD_COLOR if on_ard else PC_COLOR
            surf.blit(small.render(name, True, TXT), (rect.x + 18, y))
            surf.blit(small.render("rodando em:", True, DIM), (rect.x + 18, y + 20))
            surf.blit(small.render("ARDUINO" if on_ard else "PC", True, col), (rect.x + 110, y + 20))

            stp = self.bench.get(name, protocol.LOC_PC)
            sta = self.bench.get(name, protocol.LOC_ARDUINO)
            cur = sta if on_ard else stp
            surf.blit(small.render(f"compute  {self._t(cur.last_compute())}", True, DIM), (rect.x + 18, y + 40))
            surf.blit(small.render(f"round-trip {self._t(cur.mean_roundtrip())}", True, DIM), (rect.x + 18, y + 58))
            surf.blit(small.render(f"chamadas/s {cur.calls_per_sec():.0f}    total {cur.total_calls}", True, DIM), (rect.x + 18, y + 76))

            # toggle (clique p/ delegar sem sair desta tela)
            tog = pygame.Rect(rect.x + 18, y + 100, 170, 32)
            pygame.draw.rect(surf, PANEL2, tog, border_radius=16)
            half = pygame.Rect(tog.x + (85 if on_ard else 0), tog.y, 85, 32)
            pygame.draw.rect(surf, col, half, border_radius=16)
            surf.blit(small.render("PC", True, BG if not on_ard else DIM), (tog.x + 30, tog.y + 8))
            surf.blit(small.render("ARDUINO", True, BG if on_ard else DIM), (tog.x + 100, tog.y + 8))
            if self.serial is not None:
                self.buttons["tog_" + name] = tog
            y += 156
        self._gauges(surf, font, small, rect)

    @staticmethod
    def _t(seconds):
        if seconds <= 0:
            return "—"
        return f"{seconds*1e6:.0f} us" if seconds < 1e-3 else f"{seconds*1e3:.2f} ms"

    def _gauges(self, surf, font, small, rect):
        """Velocímetros de processamento no rodapé do painel."""
        # tempo de IA por frame = round-trip atual de cada função delegável somado
        ms = 0.0
        fwd_cps = 0.0
        for name, _d, ready in DELEGABLE:
            if not ready:
                continue
            loc = self.router.location.get(name, protocol.LOC_PC)
            st = self.bench.get(name, loc)
            ms += st.mean_roundtrip() * 1000.0
            if name == "forward_pass":
                fwd_cps = st.calls_per_sec()
        cy = rect.bottom - 110
        self._gauge(surf, font, small, rect.x + rect.width // 2 - 80, cy, 64,
                    ms / 16.6, f"{ms:.2f} ms", "IA por frame", "(orcamento 16.6ms)")
        self._gauge(surf, font, small, rect.x + rect.width // 2 + 80, cy, 64,
                    fwd_cps / 600.0, f"{fwd_cps:.0f}/s", "forward", "chamadas/s")

    def _gauge(self, surf, font, small, cx, cy, r, frac, value, label, sub):
        import math
        frac = max(0.0, min(1.0, frac))
        box = pygame.Rect(cx - r, cy - r, 2 * r, 2 * r)
        pygame.draw.arc(surf, (55, 58, 72), box, 0, math.pi, 9)        # fundo
        col = lerp(GREEN, (230, 80, 60), frac)
        if frac > 0.01:
            pygame.draw.arc(surf, col, box, math.pi * (1 - frac), math.pi, 9)
        ang = math.pi * (1 - frac)                                     # ponteiro
        pygame.draw.line(surf, TXT, (cx, cy),
                         (cx + (r - 6) * math.cos(ang), cy - (r - 6) * math.sin(ang)), 3)
        pygame.draw.circle(surf, TXT, (cx, cy), 4)
        v = font.render(value, True, col)
        surf.blit(v, (cx - v.get_width() // 2, cy + 8))
        l = small.render(label, True, TXT)
        surf.blit(l, (cx - l.get_width() // 2, cy + 28))
        s = small.render(sub, True, DIM)
        surf.blit(s, (cx - s.get_width() // 2, cy + 44))

    # ---- TELA DELEGAÇÃO ----
    def _screen_deleg(self, surf, font, big, small):
        rect = pygame.Rect(SIDEBAR + 20, 20, W - SIDEBAR - 40, H - 40)
        self._panel(surf, rect, "Delegação de funções  (clique para mover PC <-> Arduino)")
        y = rect.y + 70
        for name, desc, ready in DELEGABLE:
            self._deleg_row(surf, font, small, rect, y, name, desc, ready)
            y += 130

    def _deleg_row(self, surf, font, small, rect, y, name, desc, ready):
        loc = self.router.location.get(name, protocol.LOC_PC)
        on_ard = loc == protocol.LOC_ARDUINO
        surf.blit(font.render(name, True, TXT), (rect.x + 30, y))
        surf.blit(small.render(desc, True, DIM), (rect.x + 30, y + 22))

        # toggle PC | ARDUINO
        tog = pygame.Rect(rect.x + 320, y, 200, 40)
        can = ready and self.serial is not None
        pygame.draw.rect(surf, PANEL2, tog, border_radius=20)
        half = pygame.Rect(tog.x + (100 if on_ard else 0), tog.y, 100, 40)
        pygame.draw.rect(surf, ARD_COLOR if on_ard else PC_COLOR, half, border_radius=20)
        surf.blit(small.render("PC", True, BG if not on_ard else DIM), (tog.x + 36, tog.y + 11))
        surf.blit(small.render("ARDUINO", True, BG if on_ard else DIM), (tog.x + 118, tog.y + 11))
        if can:
            self.buttons["tog_" + name] = tog
        else:
            lock = small.render("(só PC por enquanto)" if not ready else "(offline)", True, DIM)
            surf.blit(lock, (tog.right + 16, y + 10))

        # barras de tempo PC vs Arduino
        stp = self.bench.get(name, protocol.LOC_PC)
        sta = self.bench.get(name, protocol.LOC_ARDUINO)
        self._bar(surf, small, rect.x + 560, y, "PC", stp.mean_roundtrip(), PC_COLOR)
        self._bar(surf, small, rect.x + 560, y + 34, "Arduino", sta.mean_roundtrip(), ARD_COLOR)

    def _bar(self, surf, small, x, y, label, seconds, color):
        surf.blit(small.render(label, True, DIM), (x, y + 4))
        bx = x + 80
        full = W - SIDEBAR - 40 - (bx - SIDEBAR) - 60
        # escala log: 1us -> 0, 10ms -> full
        frac = 0.0 if seconds <= 0 else min(1.0, (np.log10(seconds * 1e6) + 0) / 4.0)
        pygame.draw.rect(surf, PANEL2, (bx, y, full, 22), border_radius=6)
        pygame.draw.rect(surf, color, (bx, y, int(full * frac), 22), border_radius=6)
        t = "—" if seconds <= 0 else (f"{seconds*1e6:.0f} us" if seconds < 1e-3 else f"{seconds*1e3:.2f} ms")
        surf.blit(small.render(t, True, TXT), (bx + full + 8, y + 3))

    # ---- helpers ----
    def _panel(self, surf, rect, title):
        pygame.draw.rect(surf, PANEL, rect, border_radius=14)
        f = pygame.font.SysFont("consolas", 18, bold=True)
        surf.blit(f.render(title, True, TXT), (rect.x + 20, rect.y + 16))

    # ---- eventos ----
    def click(self, pos):
        for sid, r in list(self.buttons.items()):
            if not r.collidepoint(pos):
                continue
            if sid in ("main", "memory", "deleg"):
                self.screen_id = sid
            elif sid == "net_learn":
                self.toggle_learning()
            elif sid.startswith("mem_"):
                self.mem_region = sid[4:]
                self.mem_base = self.monitor.net_addr if (self.mem_region == "SRAM" and self.monitor) else 0x0000
                self.mem_cache = b""
                self._mem_timer = 0.0
            elif sid == "net_human":
                self.toggle_human_mode()
            elif sid.startswith("tog_"):
                name = sid[4:]
                cur = self.router.location[name]
                to_loc = protocol.LOC_ARDUINO if cur == protocol.LOC_PC else protocol.LOC_PC
                try:
                    def sync(frm, to):
                        # funções que tocam nos pesos: sincroniza no sentido da migração
                        # p/ a IA continuar igual depois de trocar de lado.
                        if to == protocol.LOC_ARDUINO:
                            self._sync_weights()
                        else:
                            self._pull_weights()
                    needs_sync = name in ("forward_pass", "backpropagation", "atualiza_pesos")
                    self.router.move(name, to_loc, sync_state=sync if needs_sync else None)
                except Exception as ex:
                    print(f"[Delegação] Erro ao mover {name} para o Arduino: {ex}")
            break


def main():
    ap = argparse.ArgumentParser(description="Dino Neural Studio (GUI)")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=250000)
    ap.add_argument("--no-serial", action="store_true")
    args = ap.parse_args()

    sc = None
    if not args.no_serial:
        port = choose_port(args.port)
        if port:
            print(f"Conectando em {port}...")
            sc = SerialComm(port, args.baud)
            sc.open()
            if sc.request(protocol.CMD_PING, b"\x01").payload != b"\x01":
                print("PING falhou — siga em modo offline (--no-serial) ou cheque o sketch.")
                sc.close(); sc = None
        else:
            print("Sem porta — rodando offline (só PC).")

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Dino Neural Studio")
    font = pygame.font.SysFont("consolas", 17)
    big = pygame.font.SysFont("consolas", 22, bold=True)
    small = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()

    studio = Studio(sc, args.no_serial)
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                studio.click(e.pos)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_1: studio.screen_id = "main"
                elif e.key == pygame.K_2: studio.screen_id = "memory"
                elif e.key == pygame.K_3: studio.screen_id = "deleg"
                elif e.key == pygame.K_ESCAPE: running = False

        studio.step_ai()
        if studio.screen_id == "memory":
            studio.refresh_memory(dt)
        studio.draw(screen, font, big, small, clock)
        pygame.display.flip()

    if sc:
        sc.close()
    pygame.quit()


if __name__ == "__main__":
    main()
