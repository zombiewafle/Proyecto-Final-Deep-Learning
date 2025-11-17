# sr_network/discriminator.py

import torch.nn as nn

class Discriminator(nn.Module):
    """
    La red Discriminador. Es un clasificador de imágenes que aprende a
    distinguir entre imágenes reales y las generadas por el Generador.
    """
    def __init__(self, input_shape=(3, 1080, 1920)):
        super(Discriminator, self).__init__()
        
        in_channels, in_height, in_width = input_shape

        def create_block(in_filters, out_filters, first_block=False):
            layers = [
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1),
            ]
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            
            layers.extend([
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            return nn.Sequential(*layers)

        # Secuencia de bloques que reducen el tamaño de la imagen y extraen características
        self.blocks = nn.Sequential(
            create_block(in_channels, 64, first_block=True),
            create_block(64, 128),
            create_block(128, 256),
            create_block(256, 512),
        )
        
        # Clasificador final que da una puntuación de "realismo"
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, img):
        features = self.blocks(img)
        validity = self.classifier(features)
        # La salida es un tensor. No se aplica Sigmoid aquí para mayor estabilidad
        # al usar BCEWithLogitsLoss durante el entrenamiento.
        return validity