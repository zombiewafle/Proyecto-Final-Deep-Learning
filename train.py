# train.py

# --- 1. IMPORTACIONES ---
# Herramientas de PyTorch
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# Nuestras creaciones: Redes y Dataset
from sr_network.generator import Generator
from sr_network.discriminator import Discriminator
from sr_network.dataset import ImageDataset
from sr_network.loss import VGGLoss # La pérdida perceptual

# Otras utilidades
from tqdm import tqdm # Para barras de progreso bonitas
import os

# --- 2. HIPERPARÁMETROS Y CONFIGURACIÓN ---

# ¿Por qué configurar esto?
# Los hiperparámetros son las perillas que ajustamos para afinar el entrenamiento.
# Cambiar estos valores puede tener un gran impacto en el resultado.

# Rutas al dataset
HR_DIR = "dataset/HR"
LR_DIR = "dataset/LR"

# Ajustes del entrenamiento
EPOCHS = 200                  # Una "epoch" es un ciclo completo a través de todo el dataset. 200 es un buen punto de partida.
BATCH_SIZE = 4                # Procesamos las imágenes en lotes pequeños. Si tienes VRAM de sobra (ej. >12GB), puedes probar 8. Si te da error de memoria, bájalo a 2 o 1.
LEARNING_RATE = 1e-4          # La "velocidad" a la que las redes aprenden. Un valor demasiado alto puede hacer que el entrenamiento sea inestable.

# Configuración del dispositivo
# Esto comprueba si tienes una GPU NVIDIA con CUDA instalado.
# Si es así, usará la GPU (mucho más rápido). Si no, usará la CPU (muy lento).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando el dispositivo: {DEVICE}")

# Creación de carpetas para guardar modelos
os.makedirs("models", exist_ok=True)
# --- 3. INICIALIZACIÓN DE MODELOS ---

    # ¿Por qué inicializar aquí?
    # Creamos los objetos de nuestras redes y los movemos al dispositivo correcto (GPU o CPU).
def train(): 
    gen = Generator(n_residual_blocks=16).to(DEVICE)
    disc = Discriminator(input_shape=(3, 720, 1920)).to(DEVICE) # HR shape

    # --- 4. FUNCIONES DE PÉRDIDA ---

    # ¿Por qué tantas pérdidas?
    # Una GAN sofisticada no se guía por un solo objetivo. Usamos una combinación
    # para equilibrar la precisión de los píxeles con el realismo visual.

    # 1. Pérdida Adversaria: Mide qué tan bien el Generador engaña al Discriminador.
    #    Usamos BCEWithLogitsLoss porque es numéricamente más estable que usar
    #    una capa Sigmoid + BCELoss por separado.
    adversarial_loss = torch.nn.BCEWithLogitsLoss().to(DEVICE)

    # 2. Pérdida de Contenido (L1): Mide la diferencia píxel a píxel.
    #    Fuerza al generador a crear la estructura correcta de la imagen.
    #    L1 es a menudo mejor que L2 (MSE) porque produce resultados menos borrosos.
    content_loss = torch.nn.L1Loss().to(DEVICE)

    # 3. Pérdida Perceptual (VGG): La "magia" para las texturas realistas.
    #    Compara las imágenes en un "espacio de características" de alto nivel.
    perceptual_loss = VGGLoss().to(DEVICE)

    # --- 5. OPTIMIZADORES ---

    # ¿Por qué dos optimizadores?
    # El Generador y el Discriminador son dos redes separadas que compiten.
    # Cada una necesita su propio optimizador para actualizar sus pesos de forma independiente.
    # Adam es el optimizador estándar y funciona muy bien en la mayoría de los casos.

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # --- 6. CARGADOR DE DATOS (DATALOADER) ---

    # ¿Qué hace el DataLoader?
    # Es un increíblemente útil de PyTorch. Toma nuestro `ImageDataset` y automáticamente:
    # - Agrupa las imágenes en lotes (batches).
    # - Baraja los datos en cada epoch para que las redes no memoricen el orden.
    # - Puede usar múltiples núcleos de CPU para cargar los datos en paralelo (`num_workers`).

    train_dataset = ImageDataset(hr_dir=HR_DIR, lr_dir=LR_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Antes de terminar, agreguemos una capa Tanh al final del generador
    # para asegurar que la salida esté en el rango [-1, 1], igual que nuestros datos normalizados.
    # En sr_network/generator.py, cambia la última línea de `__init__` a:
    # self.conv_final = nn.Sequential(
    #     nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4),
    #     nn.Tanh()
    # )
    # --- 7. BUCLE PRINCIPAL DE ENTRENAMIENTO ---

    for epoch in range(EPOCHS):
        # Usamos tqdm para crear una barra de progreso que nos informe del avance.
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            lr_imgs = batch["lr"].to(DEVICE)
            hr_imgs = batch["hr"].to(DEVICE)

            # === FASE 1: ENTRENAR EL DISCRIMINADOR ===
            # El objetivo del Discriminador es aprender a distinguir lo real de lo falso.
            
            opt_disc.zero_grad() # Reiniciamos los gradientes para no acumularlos.

            # 1.1: Pérdida con imágenes reales
            real_pred = disc(hr_imgs)
            # Queremos que el discriminador diga que estas son reales (etiqueta = 1)
            loss_real = adversarial_loss(real_pred, torch.ones_like(real_pred))
            
            # 1.2: Pérdida con imágenes falsas
            fake_imgs = gen(lr_imgs)
            fake_pred = disc(fake_imgs.detach()) # Usamos .detach() para que los gradientes no fluyan hacia el Generador en este paso.
            # Queremos que el discriminador diga que estas son falsas (etiqueta = 0)
            loss_fake = adversarial_loss(fake_pred, torch.zeros_like(fake_pred))

            # Pérdida total del discriminador y actualización de sus pesos
            loss_disc = (loss_real + loss_fake) / 2
            loss_disc.backward()
            opt_disc.step()


            # === FASE 2: ENTRENAR EL GENERADOR ===
            # El objetivo del Generador es crear imágenes tan buenas que engañen al Discriminador.

            opt_gen.zero_grad()

            # 2.1: Pérdida Adversaria (qué tan bien engañamos al discriminador)
            # Hacemos una nueva pasada por el discriminador (sin .detach() esta vez)
            fake_pred_for_gen = disc(fake_imgs)
            # El generador GANA si el discriminador cree que sus imágenes son reales (etiqueta = 1)
            loss_adv = adversarial_loss(fake_pred_for_gen, torch.ones_like(fake_pred_for_gen))
            
            # 2.2: Pérdida de Contenido y Perceptual
            loss_con = content_loss(fake_imgs, hr_imgs)
            loss_vgg = perceptual_loss(fake_imgs, hr_imgs)
            
            # 2.3: Pérdida total del Generador
            # ¡Los pesos (1e-3, 1, 6e-3) son clave! Le dan más importancia a las pérdidas
            # de contenido y perceptual que a la adversaria al principio.
            loss_gen = 1e-3 * loss_adv + 1.0 * loss_con + 6e-3 * loss_vgg
            
            loss_gen.backward()
            opt_gen.step()

            # Actualizamos la barra de progreso con las pérdidas actuales
            progress_bar.set_postfix(Loss_D=f"{loss_disc.item():.4f}", Loss_G=f"{loss_gen.item():.4f}")

        # --- Guardar el modelo al final de cada época ---
        # Es vital guardar los "checkpoints" para poder reanudar el entrenamiento
        # o usar el modelo entrenado más tarde.
        torch.save(gen.state_dict(), f"models/generator_epoch_{epoch+1}.pth")
        # También podrías guardar el del discriminador si quisieras reanudar el entrenamiento completo
        # torch.save(disc.state_dict(), f"models/discriminator_epoch_{epoch+1}.pth")

    print("¡Entrenamiento completado!")

# --- PUNTO DE ENTRADA DEL SCRIPT ---
if __name__ == '__main__':
    # Esta es la "puerta de entrada". El código aquí dentro solo se ejecutará
    # cuando corras "python train.py" directamente.
    train()