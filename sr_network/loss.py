import torch
import torch.nn as nn
from torchvision.models import vgg19

class VGGLoss(nn.Module):
    """
    Define la pérdida perceptual usando una red VGG19 pre-entrenada.
    """
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = vgg19(pretrained=True)

        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:18]).eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.loss = nn.L1Loss()

    def forward(self, generated_img, real_img):
        gen_features = self.feature_extractor(generated_img)
        real_features = self.feature_extractor(real_img)
        
        return self.loss(gen_features, real_features)