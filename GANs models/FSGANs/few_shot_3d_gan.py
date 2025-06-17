
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.init_size = (14, 2, 2, 2)
        self.l1 = nn.Sequential(nn.Linear(latent_dim, int(torch.prod(torch.tensor(self.init_size)))))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm3d(14),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(14, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512, 0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, 0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=(2, 2, 2)),
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128, 0.8),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=(2, 2, 2)),
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64, 0.8),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 14, kernel_size=3, stride=1, padding=1),
            nn.Upsample(size=(10, 15, 30)),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], *self.init_size)
        return self.conv_blocks(out)

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(14, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(512 * 1 * 1 * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Custom dataset
class FewShot3DDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)
        self.data = self.data.transpose(0, 2, 1)
        self.data = self.data.reshape(-1, 14, 10, 15, 30)
        self.data = (self.data - 0.5) / 0.5

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), 0

# Training loop
def train_gan(data_path, latent_dim=100, epochs=100, batch_size=16, save_interval=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FewShot3DDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    os.makedirs("generated_samples", exist_ok=True)

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            valid = torch.ones(real_imgs.size(0), 1, device=device)
            fake = torch.zeros(real_imgs.size(0), 1, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(real_imgs.size(0), latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        print(f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # if (epoch + 1) % save_interval == 0:
        #     torch.save(gen_imgs[0].detach().cpu(), f"generated_samples/sample_epoch_{epoch+1}.pt")

# Example usage:
# train_gan("your_data.npy")
