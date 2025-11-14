import pygame
import cv2
import numpy as np
from pathlib import Path
from .settings import WIDTH, HEIGHT, FPS, PLAYER_SPEED, BG_COLOR, OBSTACLE_COLOR
from .player import Player

def load_sprite_if_exists():
    """Intenta cargar sprites/player.png si existe en la carpeta raíz /sprites."""
    try:
        project_root = Path(__file__).resolve().parents[2]
        sprite_path = project_root / "sprites" / "player.png"
        if sprite_path.exists():
            return pygame.image.load(sprite_path.as_posix()).convert_alpha()
    except Exception:
        pass
    return None

def upscale_surface(surface, scale_factor=2, method="bilinear"):
    """
    Reescala una superficie pygame usando smoothscale o OpenCV.
    - method: "native" (sin reescalar), "bilinear", "bicubic", "nearest"
    """
    # Si se pide "native", no hacer nada
    if method == "native" or scale_factor == 1:
        return surface

    arr = pygame.surfarray.array3d(surface)            # Surface -> np.array
    arr = np.transpose(arr, (1, 0, 2))                 # (w,h,c) -> (h,w,c)
    h, w = arr.shape[:2]
    new_size = (int(w * scale_factor), int(h * scale_factor))

    if method == "bilinear":
        resized = cv2.resize(arr, new_size, interpolation=cv2.INTER_LINEAR)
    elif method == "bicubic":
        resized = cv2.resize(arr, new_size, interpolation=cv2.INTER_CUBIC)
    elif method == "nearest":
        resized = cv2.resize(arr, new_size, interpolation=cv2.INTER_NEAREST)
    else:
        # fallback pygame
        resized = pygame.transform.smoothscale(surface, new_size)
        return resized

    resized = np.transpose(resized, (1, 0, 2))         # (h,w,c) -> (w,h,c)
    return pygame.surfarray.make_surface(resized)


class App:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH * 2, HEIGHT * 2))
        pygame.display.set_caption("Reescalado Bilineal/Bicúbico")

        self.scene_surface = pygame.Surface((WIDTH, HEIGHT))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.method = "native"  #metodo en uso

        sprite = load_sprite_if_exists()
        self.player = Player((WIDTH // 2, HEIGHT // 2), sprite_surf=sprite)

        # Obstáculos
        self.obstacles = [
            pygame.Rect(0, HEIGHT - 30, WIDTH, 30),
            pygame.Rect(WIDTH//2 - 120, HEIGHT//2 + 60, 240, 20),
            pygame.Rect(80, HEIGHT//2 - 200, 300, 20),
            pygame.Rect(80, HEIGHT//2, 300, 20),
            pygame.Rect(500, 200 - 30, 100, 10),
            pygame.Rect(700, 400, 100, 10),
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
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self.running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_1:
                    self.method = "nearest"
                elif e.key == pygame.K_2:
                    self.method = "bilinear"
                elif e.key == pygame.K_3:
                    self.method = "bicubic"

    def _update(self, dt):
        self.player.update(dt, PLAYER_SPEED, self.obstacles)

    def _draw(self):
        self.scene_surface.fill(BG_COLOR)
        for ob in self.obstacles:
            pygame.draw.rect(self.scene_surface, OBSTACLE_COLOR, ob)
        self.player.draw(self.scene_surface)

        hud = self.font.render(f"ESC para salir | 1:nearest 2:bilinear 3:bicubic, en uso:({self.method})",
            True, (230, 230, 240))
        self.scene_surface.blit(hud, (20, 20))
        
        fps = self.clock.get_fps()
        fps_text = self.font.render(f"FPS: {fps:.1f}", True, (0, 255, 0))
        self.scene_surface.blit(fps_text, (20, 50))


        upscaled = upscale_surface(self.scene_surface, scale_factor=2, method=self.method)

        self.screen.blit(upscaled, (0, 0))
        pygame.display.flip()
