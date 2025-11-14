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

    # Tamaño destino
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

    # fallback pygame (bilinear)
    return pygame.transform.smoothscale(surface, new_size)

# -------------------- juego principal --------------------

class App:
    def __init__(self):
        pygame.init()

        # Ventana “1080p” relativa a tu escena base (aquí 2×)
        self.window_scale = 2.0
        self.window = pygame.display.set_mode((int(WIDTH*self.window_scale), int(HEIGHT*self.window_scale)))
        pygame.display.set_caption("Upscale en tiempo real")

        # Superficie de render base (baja resolución)
        self.scene_surface = pygame.Surface((WIDTH, HEIGHT))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        # Estado de reescalado en vivo
        self.method = "bilinear"   # "native" | "nearest" | "bilinear" | "bicubic" | "smooth"
        self.scale_factor = 2.0    # ajustable en vivo con teclas

        sprite = load_sprite_if_exists()
        self.player = Player((WIDTH//2, HEIGHT//2), sprite_surf=sprite)

        self.obstacles = [
            pygame.Rect(0, HEIGHT - 30, WIDTH, 30),
            pygame.Rect(WIDTH//2 - 120, HEIGHT//2 + 60, 240, 20),
            pygame.Rect(80, HEIGHT//2 - 200, 300, 20),
            pygame.Rect(600, HEIGHT//2 - 30, 100, 10),
        ]

        self.running = True

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
                    self.method = "smooth"   # bilinear de pygame
                # Factor en vivo
                elif e.key == pygame.K_RIGHTBRACKET or e.key == pygame.K_PLUS:
                    self.scale_factor = min(self.scale_factor + 0.25, 4.0)
                elif e.key == pygame.K_LEFTBRACKET or e.key == pygame.K_MINUS:
                    self.scale_factor = max(self.scale_factor - 0.25, 0.5)

    def _update(self, dt):
        self.player.update(dt, PLAYER_SPEED, self.obstacles)

    def _draw(self):
        # Dibuja la escena base
        self.scene_surface.fill(BG_COLOR)
        for ob in self.obstacles:
            pygame.draw.rect(self.scene_surface, OBSTACLE_COLOR, ob)
        self.player.draw(self.scene_surface)

        # HUD
        fps = self.clock.get_fps()
        hud_lines = [
            f"FPS: {fps:.1f}",
            f"Metodo: {self.method}",
            f"factor: {self.scale_factor:.2f}",
            "0:native  1:nearest  2:bilinear  3:bicubic  4:smooth"
        ]
        for i, line in enumerate(hud_lines):
            txt = self.font.render(line, True, (230,230,240))
            self.scene_surface.blit(txt, (20, 20 + i*22))

        # Reescalado EN VIVO
        up = upscale_surface(self.scene_surface, self.scale_factor, self.method)

        # Letterboxing/centrado si no llena la ventana
        win_w, win_h = self.window.get_size()
        up_w, up_h = up.get_size()
        x = (win_w - up_w) // 2
        y = (win_h - up_h) // 2

        self.window.fill((0,0,0))
        self.window.blit(up, (x, y))
        pygame.display.flip()
