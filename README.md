 EnhancedResCNN（增强型残差卷积神经网络）模型，融合了 卷积、残差块以及 TransformerBlock（自注意力机制）。让我们按顺序对各层功能进行简单介绍。
1. 输入层（fc1）
  - Linear(in_features=4, out_features=128): 首先将输入（原始编码数据）从维度 4 投影到一个 128 维的向量空间。
  - BatchNorm1d(128): 对 128 维向量做批标准化（Batch Normalization），加速收敛并稳定训练。
  - ReLU(inplace=True): 使用 ReLU 作为激活函数。
  - Dropout(p=0.1): 在训练时随机丢弃 10% 的神经元输出，防止过拟合。
2. Conv1 层
  - Conv1d(128, 128, kernel_size=3, padding=1): 将上一层输出（形状 [batch_size, 128]）在 unsqueeze(2) 之后变为 [batch_size, 128, 1]，然后用 1D 卷积进行特征提取。
  - BatchNorm1d(128) + ReLU + Dropout(p=0.1): 继续做批标准化、激活和随机丢弃，和前面作用类似。
3. layers（Sequential 容器）
  - 包含多次 ResidualBlock 和 TransformerBlock 的交替堆叠：
    - ResidualBlock:
      - 两层 1D 卷积 (Conv1d) + 批标准化 + ReLU + Dropout，最后将输入（residual）与输出相加，实现残差连接。
      - 作用：让网络可以更深地堆叠卷积层而不会出现严重的梯度消失。
    - TransformerBlock:
      - MultiheadAttention: 自注意力机制，能够捕捉长程依赖关系。
      - LayerNorm + Dropout: 做归一化和随机丢弃。
      - FeedForward(两层全连接 + ReLU): 进一步映射和非线性变换。
      - 自注意力 + 残差连接（x = x + ...）+ LayerNorm 等结构，使网络具备 Transformer 的建模能力。
  - 您的 layers 总共有 12 个子结构（0~11），每两个子结构是一个 ResidualBlock + TransformerBlock 的组合，共 6 组。
4. Conv2 层
  - Conv1d(128, 128, kernel_size=3, padding=1): 在经过所有残差和Transformer块之后，再来一次 1D 卷积对特征进行整合。
  - 同时又有 BatchNorm1d(128) + ReLU + Dropout(p=0.1)。
5. 输出层（fc2）
  - Linear(in_features=128, out_features=100): 最终将 128 维的特征映射到长度为 100 的输出向量。
  - 假设这里的 100 就是您需要的光谱波段数、或者其他回归输出维度。
整体来说，数据流向是：
(1) 输入 (4 维) → fc1(投影到128) → [BN, ReLU, Dropout] → unsqueeze → Conv1 → BN, ReLU, Dropout →
[ResidualBlock + TransformerBlock, 堆叠若干次] → Conv2 → BN, ReLU, Dropout → fc2(输出100维)

可视化“：
            ┌───────────────┐
 (input) →  │   fc1 (4→128)  │
            └───────────────┘
                   ↓
          [BN, ReLU, Dropout]
                   ↓
         unsqueeze(2) → shape: [B, 128, 1]
                   ↓
┌───────────────────────────────┐
│      Conv1(128→128, kernel=3) │
└───────────────────────────────┘
                   ↓
          [BN, ReLU, Dropout]
                   ↓
┌────────────────────────────────────────────────────────────┐
│  layers: Sequential(                                     │
│   ┌─────────────────┐    ┌─────────────────────────────┐ │
│   │ ResidualBlock   │ -> │ TransformerBlock           │ │
│   └─────────────────┘    └─────────────────────────────┘ │
│   ┌─────────────────┐    ┌─────────────────────────────┐ │
│   │ ResidualBlock   │ -> │ TransformerBlock           │ │
│   └─────────────────┘    └─────────────────────────────┘ │
│   ...... 共6组 (12个sub-layer) ......                      │
└────────────────────────────────────────────────────────────┘
                   ↓
       Conv2(128→128, kernel=3)
                   ↓
         [BN, ReLU, Dropout]
                   ↓
           fc2(128→100)
                   ↓
              (output)
fc1：将输入从 4 维映射到 128 维。
Conv1：对 (batch_size, 128, 1) 的张量进行卷积。
ResidualBlock + TransformerBlock 交替堆叠 6 次（共 12 个小模块），让网络既能利用卷积提取局部特征，又能通过自注意力捕捉全局依赖。
Conv2：再做一次 1D 卷积融合特征。
fc2：最后输出 100 维（如光谱波段数）

