import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 超参数设置
batch_size = 64
latent_dim = 100
epochs = 20
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]
])

# 加载FashionMNIST数据集 [[2]][[8]]
dataset = torchvision.datasets.FashionMNIST(
    root='../data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True)

# 生成器网络 [[1]][[6]]
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model  = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),  # 输入1x1 → 输出4x4
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 4x4 → 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 8x8 → 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # 新增层：16x16 → 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),     # 32x32 → 64x64
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x) 

# 判别器网络 [[1]][[6]]
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model  = nn.Sequential(
            # 输入64x64 → 输出32x32 
            nn.Conv2d(1, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 → 16x16 
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 → 8x8 
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 关键修正：8x8 → 1x1 
            nn.Conv2d(512, 1, 8, 1, 0, bias=False),  # kernel_size=8 
            nn.Sigmoid()
        )
 
    def forward(self, x):
        return self.model(x).view(-1)   # 输出形状 [batch_size]

# 标准权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv')  != -1:
        nn.init.normal_(m.weight.data,  0.0, 0.02)
    elif classname.find('BatchNorm')  != -1:
        nn.init.normal_(m.weight.data,  1.0, 0.02)
        nn.init.constant_(m.bias.data,  0)

netG = Generator().to(device)
netD = Discriminator().to(device)
netG.apply(weights_init)   # 初始化生成器
netD.apply(weights_init)   # 初始化判别器

# 初始化模型和优化器 [[3]][[9]]
netG = Generator().to(device)
netD = Discriminator().to(device)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
criterion = nn.BCELoss()

# 固定噪声用于可视化生成过程
fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

# 训练循环 [[4]][[8]]
G_losses = []
D_losses = []
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        real_label = torch.full((batch_size,),  0.9, device=device)  # 原始值为1.0
        fake_label = torch.full((batch_size,),  0.1, device=device)  # 原始值为0.0

        # 训练判别器
        netD.zero_grad()
        output = netD(real_images).view(-1)
        errD_real = criterion(output, real_label)
        errD_real.backward()

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake = netG(noise)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, fake_label)
        errD_fake.backward()
        optimizerD.step()

        # 训练生成器
        netG.zero_grad()
        output = netD(fake).view(-1)
        errG = criterion(output, real_label)
        errG.backward()
        optimizerG.step()

        # 记录损失
        G_losses.append(errG.item())
        D_losses.append((errD_real + errD_fake).item())

    # 打印进度
    print(f"[Epoch {epoch+1}/{epochs}] "
          f"Loss_D: {np.mean(D_losses[-len(dataloader):]):.4f} "
          f"Loss_G: {np.mean(G_losses[-len(dataloader):]):.4f}")

# 绘制损失曲线 [[4]][[8]]
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 生成图像 [[3]][[9]]
with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(
    fake, padding=2, normalize=True), (1,2,0)))
plt.show()