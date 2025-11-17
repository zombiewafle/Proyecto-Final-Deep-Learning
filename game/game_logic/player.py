import pygame
import random
from .settings import WIDTH, HEIGHT

class Player:
    def __init__(self, pos, sprite_surf=None):
        if sprite_surf is None:
            self.surf = pygame.Surface((60, 40), pygame.SRCALPHA)
            self.surf.fill((30, 200, 255))
        else:
            self.surf = sprite_surf
        self.rect = self.surf.get_rect(center=pos)

        self.vel_y = 0.0
        self.on_ground = False

        self.ai_enabled = False
        self.ai_dir = 1              
        self.ai_timer = 0.0
        self.ai_period = 2.0         
        self.ai_jump_cooldown = 0.0  

    def set_ai(self, enabled: bool):
        self.ai_enabled = enabled

    def _axis_input(self):
        keys = pygame.key.get_pressed()
        vx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
        vy = (keys[pygame.K_DOWN]  - keys[pygame.K_UP])  
        jump = keys[pygame.K_SPACE]
        return vx, jump

    def _apply_gravity_and_vertical(self, dt, obstacles):
        gravity = 1500.0
        self.vel_y += gravity * dt
        self.rect.y += int(self.vel_y * dt)

        self.on_ground = False
        for ob in obstacles:
            if self.rect.colliderect(ob):
                if self.vel_y > 0:      # cayendo
                    self.rect.bottom = ob.top
                    self.vel_y = 0.0
                    self.on_ground = True
                elif self.vel_y < 0:    # subiendo y pega contra el techo
                    self.rect.top = ob.bottom
                    self.vel_y = 0.0

    def _move_horizontal(self, vx, dt, speed, obstacles):
        self.rect.x += int(vx * speed * dt)
        for ob in obstacles:
            if self.rect.colliderect(ob):
                if vx > 0: self.rect.right = ob.left
                elif vx < 0: self.rect.left = ob.right
                return True  # colisoin en el eje x
        return False

    def update(self, dt, speed, obstacles):
        """Control manual con teclado."""
        vx, jump = self._axis_input()

        collided_x = self._move_horizontal(vx, dt, speed, obstacles)

        # salto 
        if jump and self.on_ground:
            self.vel_y = -600.0
            self.on_ground = False

        self._apply_gravity_and_vertical(dt, obstacles)

    def update_ai(self, dt, speed, obstacles):
        """Movimiento automático sencillo (patrulla + salto ante obstáculos)."""
        
        if self.ai_jump_cooldown > 0.0:
            self.ai_jump_cooldown -= dt

        # cambia de dirección cada cierto tiempo aleatoriamente 
        self.ai_timer += dt
        if self.ai_timer >= self.ai_period:
            self.ai_timer = 0.0
            
            if random.random() < 0.4: #cambio de direccion al azar
                self.ai_dir *= -1


        # Rebotar en los bordes de pantalla
        if self.rect.left <= 0:
            self.rect.left = 0
            self.ai_dir = 1
        elif self.rect.right >= WIDTH:
            self.rect.right = WIDTH
            self.ai_dir = -1


        collided_x = self._move_horizontal(self.ai_dir, dt, speed, obstacles)

        # detección de borde
        ahead_offset = 20 * self.ai_dir
        probe = self.rect.move(ahead_offset, 2)
        is_edge = True
        for ob in obstacles:
            
            if probe.move(0, 10).colliderect(ob):
                is_edge = False
                break

        if (collided_x or is_edge) and self.on_ground and self.ai_jump_cooldown <= 0.0:
            
            #salto para no caer
            self.vel_y = -600.0
            self.on_ground = False
            self.ai_jump_cooldown = 0.6  
        elif collided_x and not self.on_ground:
            #si choca con algo, dar la vuelta
            self.ai_dir *= -1

        self._apply_gravity_and_vertical(dt, obstacles)

    def draw(self, screen):
        screen.blit(self.surf, self.rect)
