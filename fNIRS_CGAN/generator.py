import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, label_dim=3):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 7680),  # 64*40*3
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1,1)),  # (32,80,5)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1,1)),  # (16,160,10)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # (1,160,10)
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([noise, label_embedding], dim=1)
        x = self.fc(x)
        x = x.view(-1, 64, 40, 3)
        x = self.deconv(x)
        return x  # (batch, 1, 160, 10) 