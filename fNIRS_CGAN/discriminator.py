import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, label_dim=3):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(1 + label_dim, 16, kernel_size=3, stride=2, padding=1),  # (16,80,5)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32,40,3)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64,20,2)
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 20 * 2, 320),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(320, 1)
        )

    def forward(self, x, labels):
        # x: (batch, 1, 160, 10)
        label_embedding = self.label_emb(labels).view(labels.size(0), -1, 1, 1)
        label_map = label_embedding.repeat(1, 1, x.size(2), x.size(3))  # (batch, label_dim, 160, 10)
        x = torch.cat([x, label_map], dim=1)  # (batch, 1+label_dim, 160, 10)
        x = self.conv(x)
        out = self.fc(x)
        return out  # (batch, 1) 