# sr_network/loss.py
import torch
import torch.nn as nn
from torchvision.models import vgg19

class VGGLoss(nn.Module):
    """
    Define la pérdida perceptual usando una red VGG19 pre-entrenada.
    """
    def __init__(self):
        super(VGGLoss, self).__init__()
        # 1. Cargamos el modelo VGG19 que ya fue entrenado en millones de imágenes.
        vgg = vgg19(pretrained=True)

        # 2. Nos quedamos solo con las capas que extraen características, antes
        #    de la parte de clasificación. La capa 18 es un buen punto intermedio.
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:18]).eval()
        
        # 3. Congelamos los pesos de VGG. No queremos entrenarlo, solo usarlo
        #    como un juez estático y experto.
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # 4. Usaremos la pérdida L1 para comparar las características extraídas.
        self.loss = nn.L1Loss()

    def forward(self, generated_img, real_img):
        """
        Calcula la pérdida.
        """
        # Extraemos las "impresiones" de VGG tanto de la imagen generada como de la real.
        gen_features = self.feature_extractor(generated_img)
        real_features = self.feature_extractor(real_img)
        
        # Devolvemos la diferencia entre esas impresiones.
        return self.loss(gen_features, real_features)