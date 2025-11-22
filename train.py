import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from sr_network.generator import Generator
from sr_network.discriminator import Discriminator
from sr_network.dataset import ImageDataset
from sr_network.loss import VGGLoss 

from tqdm import tqdm 
import os

HR_DIR = "HR_128"
LR_DIR = "LR_64"

# Ajustes del entrenamiento
EPOCHS = 10              
BATCH_SIZE = 2               
LEARNING_RATE = 1e-4         

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando el dispositivo: {DEVICE}")

os.makedirs("models", exist_ok=True)

def train():
    use_amp = (DEVICE == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    gen = Generator(n_residual_blocks=2).to(DEVICE)

    disc = Discriminator(input_shape=(3, 256, 256)).to(DEVICE) 

    adversarial_loss = torch.nn.BCEWithLogitsLoss().to(DEVICE)


    content_loss = torch.nn.L1Loss().to(DEVICE)


    perceptual_loss = VGGLoss().to(DEVICE)


    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    train_dataset = ImageDataset(hr_dir=HR_DIR, lr_dir=LR_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


    
    for epoch in range(EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, batch in enumerate(progress_bar):
            lr_imgs = batch["lr"].to(DEVICE)
            hr_imgs = batch["hr"].to(DEVICE)

            # red juez
            opt_disc.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                real_pred = disc(hr_imgs)
                loss_real = adversarial_loss(real_pred, torch.ones_like(real_pred))

                fake_imgs = gen(lr_imgs)
                fake_pred = disc(fake_imgs.detach())
                loss_fake = adversarial_loss(fake_pred, torch.zeros_like(fake_pred))

                loss_disc = (loss_real + loss_fake) / 2

            if use_amp:
                scaler.scale(loss_disc).backward()
                scaler.step(opt_disc)
            else:
                loss_disc.backward()
                opt_disc.step()

            # red generadora
            opt_gen.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                fake_pred_for_gen = disc(fake_imgs)
                loss_adv = adversarial_loss(fake_pred_for_gen, torch.ones_like(fake_pred_for_gen))
                loss_con = content_loss(fake_imgs, hr_imgs)
                loss_vgg = perceptual_loss(fake_imgs, hr_imgs)

                loss_gen = 1e-3 * loss_adv + 1.0 * loss_con + 6e-3 * loss_vgg

            if use_amp:
                scaler.scale(loss_gen).backward()
                scaler.step(opt_gen)
                scaler.update()
            else:
                loss_gen.backward()
                opt_gen.step()

            progress_bar.set_postfix(
                Loss_D=f"{loss_disc.item():.4f}",
                Loss_G=f"{loss_gen.item():.4f}"
            )


        
            torch.save(gen.state_dict(), f"models/generator_epoch_{epoch+1}.pth")
        
    print("entrenamiento completado")


if __name__ == '__main__':
    train()