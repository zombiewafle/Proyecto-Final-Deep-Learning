# sr_network/dataset.py
# ese script es el que nos va a servir de puente entre las carpetas de imagenes y pytorch. 
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

class ImageDataset(Dataset):
    """
    Clase para cargar los pares de imágenes LR y HR.
    """
    def __init__(self, hr_dir, lr_dir):
        self.hr_files = sorted(glob.glob(hr_dir + "/*.png"))
        self.lr_files = sorted(glob.glob(lr_dir + "/*.png"))
        
        # Normalizador para las imágenes (ajusta los valores de píxeles a [-1, 1])
        self.normalizer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        # Devuelve el número total de imágenes HR
        return len(self.hr_files)

    def __getitem__(self, index):
        # Carga una imagen LR y su correspondiente HR
        hr_img = cv2.imread(self.hr_files[index % len(self.hr_files)])
        lr_img = cv2.imread(self.lr_files[index % len(self.lr_files)])

        # OpenCV carga en formato BGR, lo convertimos a RGB
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

        # Convierte las imágenes a tensores de PyTorch y ajusta el rango de píxeles a [0, 1]
        hr_img = torch.from_numpy(hr_img.astype(np.float32) / 255.0)
        lr_img = torch.from_numpy(lr_img.astype(np.float32) / 255.0)

        # Cambia el formato de (Alto, Ancho, Canales) a (Canales, Alto, Ancho)
        hr_img = hr_img.permute(2, 0, 1)
        lr_img = lr_img.permute(2, 0, 1)

        # Normaliza las imágenes a [-1, 1]
        hr_img = self.normalizer(hr_img)
        lr_img = self.normalizer(lr_img)

        return {"lr": lr_img, "hr": hr_img}