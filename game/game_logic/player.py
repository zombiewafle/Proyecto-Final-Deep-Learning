import pygame

class Player:
    #Player:
        #self
        #posicion en el mundo
        #sprite

    def __init__(self, pos, sprite_surf=None):
        if sprite_surf is None:
            self.surf = pygame.Surface((60, 40), pygame.SRCALPHA)
            self.surf.fill((30, 200, 255))
        else:
            self.surf = sprite_surf
        self.rect = self.surf.get_rect(center=pos)

        #Variables físicas
        self.vel_y = 0           
        self.on_ground = False


    #mover personaje
    def _axis_input(self): 
        keys = pygame.key.get_pressed()
        vx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
        vy = (keys[pygame.K_DOWN]  - keys[pygame.K_UP])
        jump = keys[pygame.K_SPACE]
        return vx, vy

    def update(self, dt, speed, obstacles):
        vx, jump = self._axis_input()

        # --- Movimiento horizontal ---
        self.rect.x += int(vx * speed * dt)
        for ob in obstacles:
            if self.rect.colliderect(ob):
                if vx > 0: self.rect.right = ob.left
                elif vx < 0: self.rect.left = ob.right

        # --- Gravedad ---
        gravity = 1500  # píxeles/s²
        self.vel_y += gravity * dt

        # --- Salto ---
        if jump and self.on_ground:
            self.vel_y = -600   # impulso hacia arriba
            self.on_ground = False

        # --- Movimiento vertical ---
        self.rect.y += int(self.vel_y * dt)

        # --- Colisión con suelo/plataformas ---
        self.on_ground = False
        for ob in obstacles:
            if self.rect.colliderect(ob):
                if self.vel_y > 0:  # cayendo
                    self.rect.bottom = ob.top
                    self.vel_y = 0
                    self.on_ground = True
                elif self.vel_y < 0:  # subiendo y golpea techo
                    self.rect.top = ob.bottom
                    self.vel_y = 0


    def draw(self, screen):
        screen.blit(self.surf, self.rect)