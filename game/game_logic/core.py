import pygame
import cv2
import numpy as np
from pathlib import Path
from .settings import WIDTH, HEIGHT, FPS, PLAYER_SPEED, BG_COLOR, OBSTACLE_COLOR
from .player import Player

# -------------------- utilidades --------------------

def load_sprite_if_exists():
    try:
        project_root = Path(__file__).resolve().parents[2]
        sprite_path = project_root / "sprites" / "player.png"
        if sprite_path.exists():
            return pygame.image.load(sprite_path.as_posix()).convert_alpha()
    except Exception:
        pass
    return None

def upscale_surface(surface, scale_factor=2.0, method="bilinear"):
    """
    Reescala una Surface. Métodos:
      - "native": no reescala
      - "nearest" / "bilinear" / "bicubic" (OpenCV)
      - "smooth": pygame.transform.smoothscale (bilinear pygame)
    """
    if method == "native" or abs(scale_factor - 1.0) < 1e-6:
        return surface

    w, h = surface.get_size()
    new_size = (int(w * scale_factor), int(h * scale_factor))

    if method in ("nearest", "bilinear", "bicubic"):
        arr = pygame.surfarray.array3d(surface)      # (w,h,3)
        arr = np.transpose(arr, (1, 0, 2))           # -> (h,w,3)
        interp = {
            "nearest":  cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic":  cv2.INTER_CUBIC
        }[method]
        resized = cv2.resize(arr, new_size, interpolation=interp)
        resized = np.transpose(resized, (1, 0, 2))   # -> (w,h,3)
        return pygame.surfarray.make_surface(resized)

    return pygame.transform.smoothscale(surface, new_size)

def save_surface_png(surface, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(path))





# -------------------- juego principal --------------------

class App:
    def __init__(self):
        pygame.init()

        self.window_scale = 2.0
        self.window = pygame.display.set_mode((int(WIDTH*self.window_scale), int(HEIGHT*self.window_scale)))
        pygame.display.set_caption("Upscale en tiempo real")

        self.scene_surface = pygame.Surface((WIDTH, HEIGHT))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        # Reescalado en vivo
        self.method = "bilinear"   # "native" | "nearest" | "bilinear" | "bicubic" | "smooth"
        self.scale_factor = 2.0

        # Player
        sprite = load_sprite_if_exists()
        self.player = Player((WIDTH//2, HEIGHT//2), sprite_surf=sprite)

        # Obstáculos
                # --- MAPA DE PLATAFORMAS PARA IA ---
        ground_h = 30

        self.obstacles = []

        # Piso principal con un gran hueco en el centro
        self.obstacles.append(pygame.Rect(0, HEIGHT - ground_h, WIDTH // 3, ground_h))            # tramo izquierda
        self.obstacles.append(pygame.Rect(WIDTH // 3 + 150, HEIGHT - ground_h, WIDTH // 3, ground_h))  # tramo centro-derecha
        self.obstacles.append(pygame.Rect(2 * WIDTH // 3 + 80, HEIGHT - ground_h, WIDTH // 17, ground_h))    # tramo derecha

        # Plataformas bajas (para que la IA suba/baje)
        self.obstacles.append(pygame.Rect(50, HEIGHT - 170, 180, 20))
        self.obstacles.append(pygame.Rect(300, HEIGHT - 210, 160, 20))
        self.obstacles.append(pygame.Rect(550, HEIGHT - 150, 180, 20))

        # Plataformas más altas
        # self.obstacles.append(pygame.Rect(0, HEIGHT - 290, 150, 15))
        
        moving_platform = {
            "rect": pygame.Rect(0, HEIGHT - 290, 150, 15),
            "vel": 120,               # velocidad vertical en píxeles/s
            "min_y": HEIGHT - 350,    # límite superior
            "max_y": HEIGHT - 250,    # límite inferior
            "type": "moving"
        }

        self.obstacles.append(moving_platform)

        self.obstacles.append(pygame.Rect(500, HEIGHT - 350, 170, 20))

        # “Piso” pequeño flotante casi al centro
        self.obstacles.append(pygame.Rect(WIDTH // 2 + 370, HEIGHT - 80, 100, 20))

        # Paredes invisibles en los bordes (para que no se salga)
        wall_w = 10
        self.obstacles.append(pygame.Rect(-wall_w, 0, wall_w, HEIGHT))         # pared izquierda
        self.obstacles.append(pygame.Rect(WIDTH, 0, wall_w, HEIGHT))           # pared derecha


        # AI + grabación
        self.auto_move = False
        self.record = False
        self.frame_id = 0
        self.data_root = Path(__file__).resolve().parents[2] / "data"
        # data/lr -> baja resolución (scene)
        # data/sr_classic -> upscaling clásico (para baseline)

        self.running = True

    def _update_moving_platforms(self, dt):
        for ob in self.obstacles:
            if isinstance(ob, dict) and ob.get("type") == "moving":
                rect = ob["rect"]
                rect.y += int(ob["vel"] * dt)

                # Rebotar entre límites
                if rect.y <= ob["min_y"]:
                    rect.y = ob["min_y"]
                    ob["vel"] *= -1
                elif rect.y >= ob["max_y"]:
                    rect.y = ob["max_y"]
                    ob["vel"] *= -1

    def _extract_rects(self):
        out = []
        for ob in self.obstacles:
            if isinstance(ob, pygame.Rect):
                out.append(ob)
            elif isinstance(ob, dict):
                out.append(ob["rect"])
        return out


    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self._handle_events()
            self._update(dt)
            self._draw()
        pygame.quit()

    def _handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                # Toggle AI
                elif e.key == pygame.K_a:
                    self.auto_move = not self.auto_move
                    self.player.set_ai(self.auto_move)
                # Toggle grabación
                elif e.key == pygame.K_r:
                    self.record = not self.record
                # Métodos en vivo
                elif e.key == pygame.K_0:
                    self.method = "native"
                elif e.key == pygame.K_1:
                    self.method = "nearest"
                elif e.key == pygame.K_2:
                    self.method = "bilinear"
                elif e.key == pygame.K_3:
                    self.method = "bicubic"
                elif e.key == pygame.K_4:
                    self.method = "smooth"
                # Factor en vivo
                elif e.key in (pygame.K_RIGHTBRACKET, pygame.K_PLUS):
                    self.scale_factor = min(self.scale_factor + 0.25, 4.0)
                elif e.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                    self.scale_factor = max(self.scale_factor - 0.25, 0.5)

    def _update(self, dt):
        # primero actualizamos plataformas móviles
        self._update_moving_platforms(dt)

        # extraemos solo rectángulos para colisiones del player
        obstacle_rects = self._extract_rects()

        if self.auto_move:
            self.player.update_ai(dt, PLAYER_SPEED, obstacle_rects)
        else:
            self.player.update(dt, PLAYER_SPEED, obstacle_rects)


    def _draw(self):
        # Dibuja la escena base (LR)
        self.scene_surface.fill(BG_COLOR)
        for ob in self.obstacles:
            rect = ob["rect"] if isinstance(ob, dict) else ob
            pygame.draw.rect(self.scene_surface, OBSTACLE_COLOR, rect)

        self.player.draw(self.scene_surface)

        # HUD
        fps = self.clock.get_fps()
        hud_lines = [
            f"FPS: {fps:.1f}",
            f"Metodo: {self.method}",
            f"factor: {self.scale_factor:.2f}",
            f"A (auto_move): {self.auto_move}  |  R (record): {self.record}",
            "0:native  1:nearest  2:bilinear  3:bicubic  4:smooth   [ / ] factor"
        ]
        for i, line in enumerate(hud_lines):
            txt = self.font.render(line, True, (230,230,240))
            self.scene_surface.blit(txt, (20, 20 + i*22))

        # Reescalado EN VIVO para mostrar
        up = upscale_surface(self.scene_surface, self.scale_factor, self.method)

        # Centrado en ventana
        win_w, win_h = self.window.get_size()
        up_w, up_h = up.get_size()
        x = (win_w - up_w) // 2
        y = (win_h - up_h) // 2
        self.window.fill((0,0,0))
        self.window.blit(up, (x, y))
        pygame.display.flip()

        # --- Grabación opcional (genera dataset) ---
        if self.record:
            # Guarda LR (scene) como data/lr/000001.png
            lr_path = self.data_root / "lr" / f"{self.frame_id:06d}.png"
            save_surface_png(self.scene_surface, lr_path)

            # Genera baseline clásico (por defecto bicubico) y guarda en sr_classic
            sr_classic = upscale_surface(self.scene_surface, self.scale_factor, "bicubic")
            sr_path = self.data_root / "sr_classic" / f"{self.frame_id:06d}.png"
            save_surface_png(sr_classic, sr_path)

            self.frame_id += 1
