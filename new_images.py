import os
from PIL import Image

# === Carpetas ===
HR_ORIGINAL_DIR = "/home/zombiewafle/Documentos/GitHub/Proyecto-Final-Deep-Learning/dataset/HR_NEW"         
HR_128_DIR = "HR_128"           
LR_64_DIR = "LR_64"             

os.makedirs(HR_128_DIR, exist_ok=True)
os.makedirs(LR_64_DIR, exist_ok=True)

# === Configuración ===
HR_SIZE = (128, 128)            
UPSCALE_FACTOR = 2              
LR_SIZE = (HR_SIZE[0] // UPSCALE_FACTOR, HR_SIZE[1] // UPSCALE_FACTOR)

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

print("=== Generando HR_128 y LR_64 ===")

for fname in os.listdir(HR_ORIGINAL_DIR):
    if not fname.lower().endswith(VALID_EXT):
        continue

    src = os.path.join(HR_ORIGINAL_DIR, fname)
    hr_dst = os.path.join(HR_128_DIR, fname)
    lr_dst = os.path.join(LR_64_DIR, fname)

    with Image.open(src) as img:
        img = img.convert("RGB")

        hr_img = img.resize(HR_SIZE, Image.BICUBIC)
        hr_img.save(hr_dst)

        lr_img = hr_img.resize(LR_SIZE, Image.BICUBIC)
        lr_img.save(lr_dst)

print("HR_128 generado.")
print("LR_64 generado.")
print("Proceso completado con éxito.")
