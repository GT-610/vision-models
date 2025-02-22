import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn

import os
file_dir = os.path.dirname(__file__) # Locate the current directory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
latent_dim = 20
input_dim = 28*28
num_classes = 10
batch_size = 128
epochs = 10
lr = 1e-3

# 数据集处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Training set
train_dataset = torchvision.datasets.MNIST(
    os.path.join(file_dir, "../data"),
    train=True,
    download=True,
    transform=transform,
    target_transform=lambda y: torch.eye(num_classes)[y],
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=lambda batch: (
        torch.stack([x[0]  for x in batch]),
        torch.stack([x[1]  for x in batch])  # one-hot encoding
    ),
    pin_memory=True  # GPU memory optimization
)

# 模型定义
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2*latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid() # Output the probability
        )

    # Reparameterize using "ci = exp(theta) * ei + mi"
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return std * eps + mu

    def forward(self, x, y):
        # Concatenate x and y
        x_cond = torch.cat([x, y], dim=1)

        # Encode
        h = self.encoder(x_cond)
        mu, log_var = torch.chunk(h, 2, dim=1)

        # Sample
        z = self.reparameterize(mu, log_var)

        # Decode
        z_cond = torch.cat([z, y], dim=1)
        x_recon = self.decoder(z_cond)
        return x_recon, mu, log_var

    def loss_function(self, x_recon, x, mu, log_var):
        # Reconstruction loss using binary cross entropy
        bce_loss = nn.BCELoss(reduction='sum')(x_recon, x)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return bce_loss + kl_loss

model = CVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train():
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            labels = labels.to(device)

            x_recon, mu, log_var = model(data, labels)
            loss = model.loss_function(x_recon, data, mu, log_var)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader.dataset)}")

def generate_conditional_image(label, n=20):
    model.eval()
    with torch.no_grad():
        y = torch.eye(num_classes)[label].repeat(n, 1).to(device)
        z = torch.randn(n, latent_dim, device=device)
        z_cond = torch.cat([z, y], dim=1)
        generated_images = model.decoder(z_cond).view(-1, 1, 28, 28)

    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(4, 5, i+1)
        plt.imshow(generated_images[i].cpu().numpy().reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
starter.record()
train()
generate_conditional_image(6)
ender.record()
torch.cuda.synchronize()
print(f'GPU Time: {starter.elapsed_time(ender)/1000:.2f}s')