import torch 
import torch.nn  as nn 
import torch.optim  as optim 
import torchvision 
import torchvision.transforms  as transforms 
import matplotlib.pyplot  as plt 
import numpy as np 
 
import os
file_dir = os.path.dirname(__file__)

# 设备配置 
device = torch.device("cuda"  if torch.cuda.is_available()  else "cpu")

# 超参数 
latent_dim = 100    # 噪声向量维度 
img_shape = (1, 28, 28)  # MNIST图像尺寸 
batch_size = 128 
epochs = 100       # 建议训练轮次 
lr = 1e-3         # 学习率 

# 数据预处理（-1~1归一化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 单通道MNIST 
])
 
# 加载数据集 
dataset = torchvision.datasets.MNIST( 
    os.path.join(file_dir, "../data"),
    train=True, download=True, transform=transform 
)
dataloader = torch.utils.data.DataLoader( 
    dataset, batch_size=batch_size, shuffle=True 
)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model  = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()  # 输出归一化到[-1,1]
        )
 
    def forward(self, z):
        img = self.model(z) 
        return img.view(img.size(0),  *img_shape)
 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model  = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出概率值 
        )
 
    def forward(self, img):
        img_flat = img.view(img.size(0),  -1)
        validity = self.model(img_flat) 
        return validity

# 初始化网络 
generator = Generator().to(device)
discriminator = Discriminator().to(device)
 
# 定义损失函数与优化器 
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(),  lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(),  lr=lr)

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # 数据准备 
        real_imgs = imgs.to(device) 
        valid_label = torch.ones(imgs.size(0),  1).to(device)
        fake_label = torch.zeros(imgs.size(0),  1).to(device)
 
        # --------------------- 
        #  训练判别器 
        # --------------------- 
        optimizer_D.zero_grad() 
 
        # 真实样本损失 
        real_loss = adversarial_loss(discriminator(real_imgs), valid_label)
 
        # 生成样本损失 
        z = torch.randn(imgs.size(0),  latent_dim).to(device)
        fake_imgs = generator(z)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()),  fake_label)
 
        # 反向传播 
        d_loss = (real_loss + fake_loss) / 2 
        d_loss.backward() 
        optimizer_D.step() 
 
        # --------------------- 
        #  训练生成器 
        # --------------------- 
        optimizer_G.zero_grad() 
 
        # 生成器目标：让判别器认为生成样本为真 
        g_loss = adversarial_loss(discriminator(fake_imgs), valid_label)
 
        g_loss.backward() 
        optimizer_G.step() 
 
        # 打印训练状态 
        if i % 200 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}") 
def generate_samples(grid_size=5):
    generator.eval() 
    with torch.no_grad(): 
        z = torch.randn(grid_size**2,  latent_dim).to(device)
        gen_imgs = generator(z).cpu()
 
        fig, axs = plt.subplots(grid_size,  grid_size, figsize=(10,10))
        cnt = 0 
        for i in range(grid_size):
            for j in range(grid_size):
                axs[i,j].imshow(gen_imgs[cnt,0].numpy(), cmap='gray')
                axs[i,j].axis('off')
                cnt += 1 
        plt.show() 
 
generate_samples()
