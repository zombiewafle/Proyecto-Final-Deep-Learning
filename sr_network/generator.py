# sr_network/generator.py

import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Bloque Residual: El componente fundamental que permite a la red
    aprender detalles muy finos sin perder información.
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),  
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    """
    La red Generador completa.
    Toma una imagen LR y produce una imagen SR.
    """
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16, upscale_factor=2):
        super(Generator, self).__init__()

        # 1. Capa inicial: extrae características básicas de la imagen LR
        self.conv_initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), 
            nn.PReLU()
        )

        # 2. Cuerpo Principal: 16 bloques residuales para el procesamiento profundo
        residual_blocks = [ResidualBlock(in_channels=64) for _ in range(n_residual_blocks)]
        self.body = nn.Sequential(*residual_blocks)

        # 3. Capa post-residual: Combina las características de bajo y alto nivel
        self.conv_post_residual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64)
        )

        # 4. Módulo de escalado: Amplía la imagen de forma inteligente
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor), # Reorganiza píxeles para aumentar resolución
            nn.PReLU(),
        )
        
        # 5. Capa de salida: Convierte las características a una imagen RGB Y asegura el rango [-1, 1]
        self.conv_final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4),
            nn.Tanh()  
        )


    def forward(self, x):
        out_initial = self.conv_initial(x)
        out_residual = self.body(out_initial)
        out_post = self.conv_post_residual(out_residual)
        
        out = self.upsample(out_initial + out_post)
        out = self.conv_final(out)
        return out