import pygame
import cv2
import numpy as np
from pathlib import Path

import os

import torch
from sr_network.generator import Generator

from game_logic.settings import WIDTH, HEIGHT, FPS, PLAYER_SPEED, BG_COLOR, OBSTACLE_COLOR
from game_logic.player import Player

import csv

def load_sprite_if_exists():
    try:
        project_root = Path(__file__).resolve().parents[2]
        sprite_path = project_root / "sprites" / "player.png"
        if sprite_path.exists():
            return pygame.image.load(sprite_path.as_posix()).convert_alpha()
    except Exception:
        pass
    return None


def surface_to_tensor(surface, device):
    """
    Convierte una pygame.Surface en un tensor [1,3,H,W] en [0,1].
    """
    arr = pygame.surfarray.array3d(surface)      
    arr = np.transpose(arr, (1, 0, 2))           
    img = torch.from_numpy(arr).float() / 255.0  
    img = img.permute(2, 0, 1).unsqueeze(0)      
    return img.to(device)


def tensor_to_surface(tensor):
    """
    Convierte un tensor [1,3,H,W] o [3,H,W] en pygame.Surface.
    Asume rango [0,1].
    """
    if tensor.dim() == 4:
        img = tensor.squeeze(0)
    else:
        img = tensor
    img = img.permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()  
    img = (img * 255.0).astype(np.uint8)
    img = np.transpose(img, (1, 0, 2))  
    return pygame.surfarray.make_surface(img)


def sr_upscale_surface(surface, model, device):
    """
    Aplica super-resolución usando la red (Generator).
    Asume que el modelo fue entrenado para un factor fijo
    """
    model.eval()
    with torch.no_grad():
        inp = surface_to_tensor(surface, device)   
        out = model(inp)                           
    return tensor_to_surface(out)


def classic_upscale_surface(surface, scale_factor=2.0, method="bilinear"):
    if method == "native" or abs(scale_factor - 1.0) < 1e-6:
        return surface

    w, h = surface.get_size()
    new_size = (int(w * scale_factor), int(h * scale_factor))

    if method in ("nearest", "bilinear", "bicubic"):
        arr = pygame.surfarray.array3d(surface)      
        arr = np.transpose(arr, (1, 0, 2))           
        interp = {
            "nearest":  cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic":  cv2.INTER_CUBIC
        }[method]
        resized = cv2.resize(arr, new_size, interpolation=interp)
        resized = np.transpose(resized, (1, 0, 2))   
        return pygame.surfarray.make_surface(resized)

    # fallback: smoothscale de pygame
    return pygame.transform.smoothscale(surface, new_size)


def save_surface_png(surface, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(path))


class AppSR:
    def __init__(self):
        self.metrics = []
        pygame.init()
        
        pygame.display.set_caption("SR Game - Generator in the Loop")

        
        self.display_scale = 1.0
        # self.window = pygame.display.set_mode(
        #     (int(WIDTH * self.display_scale), int(HEIGHT * self.display_scale))
        # )
        self.window = pygame.display.set_mode((0, 0), pygame.NOFRAME)


        
        self.scene_surface = pygame.Surface((WIDTH, HEIGHT))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        self.method = "native"
        self.scale_factor = 2.0   

        self.sr_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SR] Usando dispositivo: {self.sr_device}")
        torch.backends.cudnn.benchmark = True

        self.sr_model = Generator(n_residual_blocks=2).to(self.sr_device)

        
        if "__file__" in globals():
        
            project_root = Path(__file__).resolve().parent
        else:
            project_root = Path(os.getcwd())

        print(f"[SR] project_root = {project_root}")

        model_path = project_root / "models" / "generator_epoch_50.pth"
        print(f"[SR] Cargando modelo desde: {model_path}")

        model_path = project_root / "models"/ "generator_epoch_50.pth"


        state_dict = torch.load(model_path, map_location=self.sr_device)
        self.sr_model.load_state_dict(state_dict)
        self.sr_model.eval()

        # Player 
        self.player_spawn_pos = (WIDTH // 2, HEIGHT // 2)

        sprite = load_sprite_if_exists()
        self.player = Player(self.player_spawn_pos, sprite_surf=sprite)

        # Obstáculos
        self.obstacles = []
        self._build_level()

        #
        self.auto_move = False
        self.record = False
        self.frame_id = 0
        self.data_root = project_root / "dataset"


        self.methods_cycle = ["native", "nearest", "bilinear", "bicubic", "smooth", "sr"]
        self.method_index = 0
        self.auto_cycle = False
        self.last_switch = 0
        self.switch_delay = 30000  


        self.running = True

    def save_metrics(self):
        with open("metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "fps"])
            writer.writeheader()
            writer.writerows(self.metrics)


    def _auto_change_method(self):
        if not self.auto_cycle:
            return

        now = pygame.time.get_ticks()

        if now - self.last_switch >= self.switch_delay:
            # Avanzar al siguiente método
            self.method_index = (self.method_index + 1) % len(self.methods_cycle)
            self.method = self.methods_cycle[self.method_index]

            print("[AUTO] Método cambiado a:", self.method)

            self.last_switch = now


    def _build_level(self):
        ground_h = 30

        # Piso principal con un hueco en el centro
        self.obstacles.append(pygame.Rect(0, HEIGHT - ground_h, WIDTH // 3, ground_h))
        self.obstacles.append(
            pygame.Rect(WIDTH // 3 + 150, HEIGHT - ground_h, WIDTH // 3, ground_h)
        )
        self.obstacles.append(
            pygame.Rect(2 * WIDTH // 3 + 80, HEIGHT - ground_h, WIDTH // 17, ground_h)
        )

        # Plataformas bajas
        self.obstacles.append(pygame.Rect(50, HEIGHT - 170, 180, 20))
        self.obstacles.append(pygame.Rect(300, HEIGHT - 210, 160, 20))
        self.obstacles.append(pygame.Rect(550, HEIGHT - 150, 180, 20))

        # Plataforma móvil
        moving_platform = {
            "rect": pygame.Rect(0, HEIGHT - 290, 150, 15),
            "vel": 120,
            "min_y": HEIGHT - 350,
            "max_y": HEIGHT - 250,
            "type": "moving",
        }
        self.obstacles.append(moving_platform)

        # Otra plataforma alta
        self.obstacles.append(pygame.Rect(500, HEIGHT - 350, 170, 20))

        # Piso flotante casi al centro
        self.obstacles.append(
            pygame.Rect(WIDTH // 2 + 370, HEIGHT - 80, 100, 20)
        )

        # Paredes invisibles laterales
        wall_w = 10
        self.obstacles.append(pygame.Rect(-wall_w, 0, wall_w, HEIGHT))
        self.obstacles.append(pygame.Rect(WIDTH, 0, wall_w, HEIGHT))

    
    def _update_moving_platforms(self, dt):
        for ob in self.obstacles:
            if isinstance(ob, dict) and ob.get("type") == "moving":
                rect = ob["rect"]
                rect.y += int(ob["vel"] * dt)

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
            dt = min(dt, 1.0 / 30.0)

            self._auto_change_method()
            
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
                elif e.key == pygame.K_a:
                    self.auto_move = not self.auto_move
                    self.player.set_ai(self.auto_move)

                    # Activar o desactivar el auto cambio de método
                    self.auto_cycle = self.auto_move

                    if self.auto_cycle:
                        self.last_switch = pygame.time.get_ticks()
                        if self.method in self.methods_cycle:
                            self.method_index = self.methods_cycle.index(self.method)

                elif e.key == pygame.K_r:
                    self.record = not self.record

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
                elif e.key == pygame.K_5:
                    self.method = "sr"

                elif e.key in (pygame.K_RIGHTBRACKET, pygame.K_PLUS):
                    self.scale_factor = min(self.scale_factor + 0.25, 4.0)
                elif e.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                    self.scale_factor = max(self.scale_factor - 0.25, 0.5)

    
    def _update(self, dt):
        self._update_moving_platforms(dt)
        obstacle_rects = self._extract_rects()

        if self.auto_move:
            self.player.update_ai(dt, PLAYER_SPEED, obstacle_rects)
        else:
            self.player.update(dt, PLAYER_SPEED, obstacle_rects)

        if self.player.rect.top > HEIGHT:
            self.player.reset(self.player_spawn_pos)

    def _draw(self):
        self.scene_surface.fill(BG_COLOR)
        for ob in self.obstacles:
            rect = ob["rect"] if isinstance(ob, dict) else ob
            pygame.draw.rect(self.scene_surface, OBSTACLE_COLOR, rect)

        self.player.draw(self.scene_surface)

        # HUD
        fps = self.clock.get_fps()


        
        self.metrics.append({
            "method": self.method,
            "fps": fps,
        })

        print("Metodo:",self.method, fps)
        self.save_metrics()

        hud_lines = [
            f"FPS: {fps:.1f}",
            f"Metodo: {self.method}",
            f"factor (clasico): {self.scale_factor:.2f}",
            f"A (auto_move): {self.auto_move}  |  R (record): {self.record}",
            "0:native  1:nearest  2:bilinear  3:bicubic  4:smooth  5:SR   [ / ] factor"
        ]
        for i, line in enumerate(hud_lines):
            txt = self.font.render(line, True, (230, 230, 240))
            self.scene_surface.blit(txt, (20, 20 + i * 22))

        if self.method == "sr":
            LR_SIZE = (64, 64)    
            HR_SIZE = (128, 128)  

            lr_small = pygame.transform.smoothscale(self.scene_surface, LR_SIZE)

            sr_small = sr_upscale_surface(lr_small, self.sr_model, self.sr_device)

            up = pygame.transform.smoothscale(sr_small, (WIDTH, HEIGHT))

        else:
            up = classic_upscale_surface(
                self.scene_surface,
                self.scale_factor,
                self.method
            )


        win_w, win_h = self.window.get_size()
        up_w, up_h = up.get_size()

        # ajuste de la ventana, manteniendo el aspect ratio
        scale = min(win_w / up_w, win_h / up_h)
        final_w = int(up_w * scale)
        final_h = int(up_h * scale)
        final_surface = pygame.transform.smoothscale(up, (final_w, final_h))

        self.window.fill((0, 0, 0))
        x = (win_w - final_w) // 2
        y = (win_h - final_h) // 2
        self.window.blit(final_surface, (x, y))
        pygame.display.flip()

        if self.record:
            hr_path = self.data_root / "HR_NEW" / f"{self.frame_id:06d}.png"
            save_surface_png(self.scene_surface, hr_path)

            lr_img = classic_upscale_surface(self.scene_surface, 0.5, "bicubic")
            lr_path = self.data_root / "LR_NEW" / f"{self.frame_id:06d}.png"
            save_surface_png(lr_img, lr_path)

            self.frame_id += 1



if __name__ == "__main__":
    app = AppSR()
    app.run()

