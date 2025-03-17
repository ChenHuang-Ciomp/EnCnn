# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:17:42 2025

@author: Administrator
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib import rcParams
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
torch.manual_seed(12046)
import numpy as np
# 设备选择，支持 GPU 加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置字体为 SimHei（黑体）
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
class ResNetBlockWithBNandDropout(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super(ResNetBlockWithBNandDropout, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批量标准化
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)  # 批量标准化
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout_prob)  # Dropout 层

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.dropout(F.relu(out))
        return out

class SpectralResNetWithBNandDropout(nn.Module):
    def __init__(self):
        super(SpectralResNetWithBNandDropout, self).__init__()
        # 第一层卷积：输入 1，输出 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # 第一层批量标准化
        
        # 残差块
        self.resblock1 = ResNetBlockWithBNandDropout(16, 32)
        self.resblock2 = ResNetBlockWithBNandDropout(32, 64)
        
        # 进行一次池化，减少特征图的尺寸
        self.pool = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 输出层：将特征映射到 (B, 100)
        self.fc = nn.Linear(64, 100)  # 因为在经过两次卷积和池化后，特征图尺寸是 8x8，所以输入全连接层的特征数是 64 * 8 * 8 = 4096
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        
        # 池化层，减少尺寸
        x = self.pool(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        
        # 输出 G_matrix
        x = self.fc(x)
        return x
# 数据增强：扩增样本数量，并进行标准化
class SpectralDataset(Dataset):
    def __init__(self, sample_path, g_matrix_path, augment_factor=10):
        data = sio.loadmat(sample_path)['sample']  # (B, 100, 100)
        labels = sio.loadmat(g_matrix_path)['G_matrix']  # (B, 100)
        
        # 数据标准化
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # 对数据进行标准化
        self.data = torch.tensor(self.scaler_x.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape), dtype=torch.float32).unsqueeze(1).to(device)  # (B, 1, 100, 100)
        self.labels = torch.tensor(self.scaler_y.fit_transform(labels), dtype=torch.float32).to(device)  # (B, 100)
        
        self.augment_factor = augment_factor  # 乘法因子以扩充数据
        
        # 生成扩增数据
        augmented_data, augmented_labels = [], []
        for i in range(augment_factor):
            scale = np.random.uniform(0.8, 1.0)  # 乘以 0.91, 0.915, 0.92 ...
            augmented_data.append(self.data * scale )
            augmented_labels.append(self.labels * scale)
        
        self.data = torch.cat(augmented_data, dim=0)
        self.labels = torch.cat(augmented_labels, dim=0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 初始化数据集
dataset = SpectralDataset('sample2D2.mat', 'G_matrix2D2.mat')

# 设置验证集比例
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# 拆分数据集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 初始化模型
model = SpectralResNetWithBNandDropout().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)  # 使用 AdamW 优化器，并添加 L2 正则化

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)

# 用于记录损失
train_losses = []
val_losses = []

# 训练和验证过程
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for data, labels in train_loader:
        optimizer.zero_grad()

        # 前向传播
        outputs = model(data)

        # 计算训练损失
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()
        # 记录训练损失
    train_losses.append(running_train_loss / len(train_loader))

    # 验证
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for data, labels in val_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
    val_losses.append(running_val_loss / len(val_loader))

    # 调整学习率
    scheduler.step(val_losses[-1])

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.6f}, Validation Loss: {val_losses[-1]:.6f}")

# 保存训练好的模型
torch.save(model.state_dict(), 'spectral_resnet_with_bn_dropout.pth')

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# 训练完后加载模型
loaded_model = SpectralResNetWithBNandDropout().to(device)
loaded_model.load_state_dict(torch.load('spectral_resnet_with_bn_dropout.pth'))
loaded_model.eval()

# 验证集上的预测
with torch.no_grad():
    predictions = []
    actuals = []
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        outputs = loaded_model(inputs)
        predictions.append(outputs.cpu().numpy())  # 确保输出张量先移到 CPU
        actuals.append(targets.cpu().numpy())  # 确保目标张量先移到 CPU

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

# 计算均方误差（MSE）、平均绝对误差（MAE）和决定系数（R²）
mse = mean_squared_error(actuals, predictions)
r2 = r2_score(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
print(f'验证集 MAE: {mae:.6f}, MSE: {mse:.6f}, R²: {r2:.6f}')

# 可视化部分样本的预测结果
num_samples_to_plot = min(5, len(predictions))
for i in range(num_samples_to_plot):
    plt.figure()
    plt.plot(actuals[i], label='实际光谱')
    plt.plot(predictions[i], label='预测光谱')
    plt.xlabel('波长索引')
    plt.ylabel('强度')
    plt.title(f'样本 {i+1} 的光谱重构')
    plt.legend()
    plt.show()


