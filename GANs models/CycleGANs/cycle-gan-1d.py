import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Residual block for 1D
class Residual1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm1d(dim)
        )
    def forward(self, x):
        return x + self.block(x)

# Generator A→B or B→A
class Generator1D(nn.Module):
    def __init__(self, channels=14, ngf=64, n_res=6):
        super().__init__()
        layers = [
            # down‐sampling
            nn.Conv1d(channels, ngf, kernel_size=7, padding=3),
            nn.InstanceNorm1d(ngf), nn.ReLU(inplace=True),
            nn.Conv1d(ngf, ngf*2,  kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(ngf*2), nn.ReLU(inplace=True),
            nn.Conv1d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(ngf*4), nn.ReLU(inplace=True),
        ]
        # residual blocks
        for _ in range(n_res):
            layers += [Residual1d(ngf*4)]
        # up‐sampling
        layers += [
            nn.ConvTranspose1d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(ngf*2), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(ngf*2, ngf,   kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm1d(ngf), nn.ReLU(inplace=True),
            nn.Conv1d(ngf, channels, kernel_size=7, padding=3),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Patch‐style Discriminator
class Discriminator1D(nn.Module):
    def __init__(self, channels=14, ndf=64):
        super().__init__()
        layers = [
            nn.Conv1d(channels, ndf,  kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        mult = 1
        for i in range(1, 4):
            layers += [
                nn.Conv1d(ndf*mult, ndf*mult*2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm1d(ndf*mult*2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            mult *= 2
        layers += [nn.Conv1d(ndf*mult, 1, kernel_size=4, padding=1)]  # output a 1D “patch”
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# loss helpers
adv_criterion   = nn.MSELoss()
cycle_criterion = nn.L1Loss()
identity_criterion = nn.L1Loss()

def get_dataloader(normal, fault, batch_size=8):
    # normal,fault shape: (n_samples,4500,14)
    A = torch.tensor(normal, dtype=torch.float).permute(0,2,1)
    B = torch.tensor(fault, dtype=torch.float).permute(0,2,1)
    ds = TensorDataset(A, B)
    return DataLoader(ds, batch_size, shuffle=True, drop_last=True)

def train_cycle_gan(normal, fault, device):
    # networks
    G_AB = Generator1D().to(device)
    G_BA = Generator1D().to(device)
    D_A  = Discriminator1D().to(device)
    D_B  = Discriminator1D().to(device)

    opt_G = optim.Adam(list(G_AB.parameters())+list(G_BA.parameters()), lr=2e-4, betas=(0.5,0.999))
    opt_DA= optim.Adam(D_A.parameters(), lr=2e-4, betas=(0.5,0.999))
    opt_DB= optim.Adam(D_B.parameters(), lr=2e-4, betas=(0.5,0.999))

    loader = get_dataloader(normal, fault, batch_size=4)
    real_label = 1.0
    fake_label = 0.0

    for epoch in range(1, 101):
        for real_A, real_B in loader:
            real_A, real_B = real_A.to(device), real_B.to(device)
            # -- Train Generators --
            opt_G.zero_grad()
            # Identity
            idt_B = G_AB(real_B)
            idt_A = G_BA(real_A)
            loss_idt = identity_criterion(idt_B, real_B)*5.0 + identity_criterion(idt_A, real_A)*5.0
            # GAN loss
            fake_B = G_AB(real_A)
            pred_fake_B = D_B(fake_B)
            loss_GAN_AB = adv_criterion(pred_fake_B, torch.ones_like(pred_fake_B))
            fake_A = G_BA(real_B)
            pred_fake_A = D_A(fake_A)
            loss_GAN_BA = adv_criterion(pred_fake_A, torch.ones_like(pred_fake_A))
            # Cycle
            rec_A = G_BA(fake_B)
            rec_B = G_AB(fake_A)
            loss_cycle = cycle_criterion(rec_A, real_A)*10.0 + cycle_criterion(rec_B, real_B)*10.0
            # total
            loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle + loss_idt
            loss_G.backward()
            opt_G.step()

            # -- Train D_A --
            opt_DA.zero_grad()
            pred_real_A = D_A(real_A)
            loss_D_real = adv_criterion(pred_real_A, torch.ones_like(pred_real_A))
            pred_fake_A = D_A(fake_A.detach())
            loss_D_fake = adv_criterion(pred_fake_A, torch.zeros_like(pred_fake_A))
            (loss_D_real + loss_D_fake).backward()
            opt_DA.step()

            # -- Train D_B --
            opt_DB.zero_grad()
            pred_real_B = D_B(real_B)
            loss_D_real = adv_criterion(pred_real_B, torch.ones_like(pred_real_B))
            pred_fake_B = D_B(fake_B.detach())
            loss_D_fake = adv_criterion(pred_fake_B, torch.zeros_like(pred_fake_B))
            (loss_D_real + loss_D_fake).backward()
            opt_DB.step()

        print(f"Epoch {epoch:03d} | G: {loss_G.item():.4f} D_A: {(loss_D_real+loss_D_fake).item():.4f} D_B: {(loss_D_real+loss_D_fake).item():.4f}")

    return G_AB, G_BA