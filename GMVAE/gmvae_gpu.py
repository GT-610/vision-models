import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional  as F 

import os
file_dir = os.path.dirname(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
latent_dim = 20
input_dim = 28*28
num_components = 10
batch_size = 128
epochs = 30
lr = 1e-3

# 数据集处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
# 训练集
train_dataset = torchvision.datasets.MNIST(
    os.path.join(file_dir, "../data"),
    train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    pin_memory=True  # GPU memory optimization
)

class GMVAE(nn.Module):
    def __init__(self):
        super(GMVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # 隐变量
        self.z_net = nn.Linear(512, latent_dim*2)
        # 混合系数
        self.w_net = nn.Linear(512, num_components)

        # 隐变量的均值和方差
        self.mu_components = nn.Parameter(torch.randn(num_components, latent_dim))
        self.logvar_components = nn.Parameter(torch.zeros(num_components, latent_dim))

        self.decoder  = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),        # 新增层 
            nn.ReLU(),
            nn.Linear(512, 1024),       # 扩展通道 
            nn.ReLU(),
            nn.Linear(1024, input_dim), # 最终输出层 
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码
        h = self.encoder(x)

        # 隐变量
        z_params = self.z_net(h)
        z_mu, z_logvar = torch.chunk(z_params, 2, dim=1)
        z = self.reparameterize(z_mu, z_logvar)

        # 混合系数
        w_logits = self.w_net(h)
        return z_mu, z_logvar, z, w_logits

    def sample_from_prior(self, num_samples, component_idx=None):
        if component_idx is None:
            # 随机选择一个混合系数
            p = torch.ones(num_components) / num_components
            component_idx = torch.multinomial(p, num_samples, replacement=True)
        
        mu = self.mu_components[component_idx]
        logvar = self.logvar_components[component_idx]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), component_idx

    def loss_function(self, x_recon, x, z_mu, z_logvar, w_logits):
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

        # 广播计算 KL 值
        kl_z = 0.5 * torch.sum( 
            self.logvar_components.unsqueeze(0)  - z_logvar.unsqueeze(1)  
            + (z_logvar.unsqueeze(1).exp()  + (z_mu.unsqueeze(1)  - self.mu_components.unsqueeze(0))**2) 
            / self.logvar_components.unsqueeze(0).exp() 
            - 1,
            dim=2 
        )

        # 混合权重KL：q(w|x) vs p(w)
        q_w = F.softmax(w_logits,  dim=1)
        log_q_w = F.log_softmax(w_logits,  dim=1)
        log_p_w = torch.log(torch.tensor(1.0/num_components,  device=x.device)) 
        kl_w = torch.sum(q_w  * (log_q_w - log_p_w), dim=1).sum()
        
        # 总损失 = 重构损失 + KL_z + KL_w 
        total_loss = recon_loss + torch.sum(kl_z  * q_w) + kl_w 
        return total_loss / x.size(0)   # 取batch平均 

model = GMVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(),  lr=lr)
 
def train():
    model.train() 
    for epoch in range(epochs):
        total_loss = 0 
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            
            # 前向传播 
            z_mu, z_logvar, z, w_logits = model(data)
            x_recon = model.decoder(z) 
            
            # 计算损失 
            loss = model.loss_function(x_recon,  data, z_mu, z_logvar, w_logits)
            
            # 反向传播 
            loss.backward() 
            optimizer.step() 
            
            total_loss += loss.item() 
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")
 
def generate_components_comparison():
    model.eval() 
    with torch.no_grad(): 
        # 生成每个成分的样本 
        fig, axs = plt.subplots(num_components,  5, figsize=(15, 30))
        for k in range(num_components):
            samples, _ = model.sample_from_prior(5,  component_idx=torch.tensor([k]*5)) 
            for i in range(5):
                axs[k][i].imshow(samples[i].view(28,28).cpu().numpy(), cmap='gray')
                axs[k][i].axis('off')
        plt.tight_layout() 
        plt.show() 
 
# 执行训练与生成
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
starter.record()
train()
generate_components_comparison()
ender.record()
torch.cuda.synchronize()
print(f'GPU Time: {starter.elapsed_time(ender)/1000:.2f}s')
