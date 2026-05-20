# OurWork1：基于 CoFE 去噪增强的行人轨迹与意图预测网络

OurWork1 是在 PTINet 基础上扩展的多任务学习框架，用于联合预测行人轨迹与穿越意图。相比于原始 PTINet，本项目引入了 **CoFE（Correction Feature Embedding）轨迹去噪修复模块**，在历史轨迹送入 LSTM-VAE 编码之前先进行去噪修复，并确保修复后位置与速度的物理一致性。

该框架同时支持 **BEV（鸟瞰图）视角**（JAAD/PIE/TITAN 数据集）和 **FPV（第一人称视角）**（T2FPV 数据集）。

---

## 模型架构

### 整体数据流

```
输入轨迹 → [CoFE 轨迹去噪] → 位置编码器(LSTMVAE)
                              → 速度重计算(一阶差分)
输入速度 → 速度编码器(LSTMVAE)
输入行为 → 行为编码器(LSTMVAE)
输入场景 → 场景编码器(LSTMVAE)   →  特征融合  →  双任务解码器
输入图像 → ConvLSTM / ResNet                                   →  速度预测
输入光流 → ResNet + LSTM                                       →  穿越意图预测
```

### 核心创新

#### 1. CoFE 轨迹去噪修复模块

CoFE 基于 GRU 编码器-解码器架构，支持 **BEV** 和 **FPV** 两种模式：

##### FPV 模式（第一人称视角）

完整实现 T2FPV 原始设计，包括：

1. **偏航角编码**：4步变换（ego相对化 → 归一化 → 弧度化 → cos/sin编码）
2. **Ego 相对距离**：计算每个 agent 相对于自车的坐标偏移
3. **ResNet 视觉融合**：可选融合 2048 维 ResNet 视觉特征
4. **帧间位移修正**：在位移空间中做 GRU 编解码
5. **绝对坐标恢复**：通过 cumsum + initial_pos 还原绝对坐标

##### BEV 模式（鸟瞰图视角）

简化版本，不依赖偏航角和 ego 相对坐标。

##### 数据流

```
绝对坐标 → 特征构建 → idxs选择 → GRU编码 → GRU解码 → 位移还原 → 修正后绝对坐标
```

#### 2. VLM 语义意图特征

CoFE 可选集成 VLM（如 CLIP）离线提取的高级语义意图特征（512 维文本嵌入）：

- 作为"语义锚点"，在 FPV 轨迹噪声极高时稳定 GRU 隐藏状态
- 通过 MLP 将 512 维意图特征映射到 GRU 隐藏空间
- 在编码步骤中与位移特征、视觉特征拼接后送入 GRU 编码器
- 在解码步骤中每个时间步参与 GRU 解码器隐藏状态更新

#### 3. 物理一致性（速度重计算）

CoFE 修复位置序列后，从修复后的位置通过一阶差分重新计算速度。这确保了位置与速度之间的物理关系始终成立。

#### 4. 向量化性能优化

CoFE 模块实现了向量化计算，显著提升性能：

| 函数 | 优化技术 | CPU 加速比 |
|------|---------|-----------|
| `ego_dists` | PyTorch gather 向量化 | **1.82x** |
| `encode_yaw` | PyTorch gather 向量化 | 1.00x |

### 多模态输入

| 输入模态 | 说明 | 编码方式 |
|---------|------|---------|
| **轨迹位置** | 历史边界框序列 (x, y, w, h) / 世界坐标 (x, y) | LSTMVAE 编码器 |
| **轨迹速度** | 帧间差分速度序列 (x, y, w, h) | LSTMVAE 编码器 |
| **行人行为** | 反应、手势、注视、点头等行为标签 | LSTMVAE 编码器 |
| **场景属性** | 交通标志，道路类型等场景特征 | LSTMVAE 编码器 |
| **行人属性** | 年龄、性别、群体大小 | MLP 编码器 |
| **视觉图像** | 场景 RGB 图像序列 | ConvLSTM / ResNet50 / ResNet18 |
| **光流** | 相邻帧稠密光流 | ResNet50 + LSTM |
| **ResNet特征** | 预提取的 ResNet 特征（FPV模式） | LSTM 编码器 |
| **VLM意图特征** | CLIP 文本嵌入（512 维） | MLP 投影（可选，用于 CoFE） |

### 双任务输出

- **速度轨迹预测**：通过 LSTMCell 解码器逐时间步生成未来速度序列
- **穿越意图预测**：通过独立的 LSTMCell 解码器输出每帧的穿越概率（二分类）

---

## 模型内部变量详解

### VAE 编码器输出变量

PTINet 使用多个 LSTMVAE 编码器分别处理不同模态的输入序列。每个 LSTMVAE 编码器的输出包含隐变量 `z` 和隐藏状态 `h`，命名规则为 `h{X}{Y}` 和 `z{X}{Y}`：

| 变量名 | 全称 | 来源 | 含义 | 形状 |
|--------|------|------|------|------|
| **hpo** | Hidden - **P**osition **O**utput | `pos_encoder` 编码器 | 位置轨迹编码器的最终 LSTM 隐藏状态（解码器初始隐藏状态的核心组成部分） | `(batch, hidden_size)` |
| **zpo** | **Z** - **P**osition **O**utput | `pos_encoder` 编码器 | 位置轨迹隐变量，对序列维度取平均后的 VAE 隐空间采样值 | `(batch, latent_size)` |
| **hsp** | Hidden - **S**peed **P**osition | `speed_encoder` 编码器 | 速度序列编码器的最终 LSTM 隐藏状态 | `(batch, hidden_size)` |
| **zsp** | **Z** - **S**peed **P**osition | `speed_encoder` 编码器 | 速度序列隐变量，对序列维度取平均 | `(batch, latent_size)` |
| **hpa** | Hidden - **P**edestrian **A**ttribute | `ped_behavior_encoder` 编码器 | 行人行为序列编码器的最终 LSTM 隐藏状态 | `(batch, hidden_size)` |
| **zpa** | **Z** - **P**edestrian **A**ttribute | `ped_behavior_encoder` 编码器 | 行人行为序列隐变量 | `(batch, latent_size)` |
| **hsa** | Hidden - **S**cene **A**ttribute | `scene_attribute_encoder` 编码器 | 场景属性序列编码器的最终 LSTM 隐藏状态 | `(batch, hidden_size)` |
| **zsa** | **Z** - **S**cene **A**ttribute | `scene_attribute_encoder` 编码器 | 场景属性序列隐变量 | `(batch, latent_size)` |
| **himg** | Hidden - **Im**a**g**e | 图像编码器（ConvLSTM） | 图像序列的最终隐藏状态（池化 + 线性映射后） | `(batch, hidden_size)` |
| **cimg** | **C**ell - **Im**a**g**e | 图像编码器（ConvLSTM） | 图像序列的最终细胞状态（池化 + 线性映射后） | `(batch, hidden_size)` |
| **himg_op** | Hidden - **Im**a**g**e **Op**tical | 光流编码器（ResNet+LSTM） | 光流序列的最终 LSTM 隐藏状态 | `(batch, hidden_size)` |
| **cimg_op** | **C**ell - **Im**a**g**e **Op**tical | 光流编码器（ResNet+LSTM） | 光流序列的最终 LSTM 细胞状态 | `(batch, hidden_size)` |
| **pb** | **P**edestrian attribute **B** | `self.mlp`（行人属性 MLP） | 行人属性（年龄、性别等）的 MLP 嵌入向量 | `(batch, hidden_size)` |

### FPV 模式专用变量（T2FPV）

FPV 模式下图像和光流的编码方式不同，使用独立的变量名：

| 变量名 | 全称 | 来源 | 含义 | 形状 |
|--------|------|------|------|------|
| **him** | Hidden - **Im**age (FPV) | `fpv_resnet_lstm`（LSTM） | FPV 模式下 ResNet 特征序列的最终 LSTM 隐藏状态 | `(batch, hidden_size)` |
| **cim** | **C**ell - **Im**age (FPV) | `fpv_resnet_lstm`（LSTM） | FPV 模式下 ResNet 特征序列的最终 LSTM 细胞状态 | `(batch, hidden_size)` |
| **hop** | Hidden - **O**ptical **P** (FPV) | 零张量占位符 | FPV 模式下光流特征占位，始终为零（当前未启用光流） | `(batch, hidden_size)` |
| **cop** | **C**ell - **O**ptical **P** (FPV) | 零张量占位符 | FPV 模式下光流细胞状态占位，始终为零（当前未启用光流） | `(batch, hidden_size)` |

### 意图特征变量

| 变量名 | 全称 | 来源 | 含义 | 形状 |
|--------|------|------|------|------|
| **intent_feature** | Intent Feature (VLM) | DataLoader batch | VLM（如 CLIP）提取的高级语义意图特征，作为"语义锚点" | `(B, intent_feature_dim)` 或 `(B, N, intent_feature_dim)` |
| **intent_emb** | Intent **Emb**edding | `self.intent_proj`（MLP） | 意图特征经 MLP 投影到 hidden_size 维度后的嵌入向量 | `(batch, hidden_size)` |

### 特征融合变量

所有模态的特征通过加法融合，得到解码器的初始状态：

| 变量名 | 全称 | 含义 | 计算方式 |
|--------|------|------|---------|
| **hds** | Hidden **D**ecoder **S**um | 解码器初始隐藏状态（融合所有模态的隐藏特征） | `hpo + hsp + hpa + hsa + pb + himg + himg_op + intent_emb`（BEV 模式）<br>`hpo + hsp + hpa + hsa + pb + him + hop + intent_emb`（FPV 模式） |
| **zds** | **Z** **D**ecoder **S**um | 解码器初始细胞状态（融合所有模态的细胞/隐变量特征） | `zpo + zsp + zpa + zsa + pb + cimg + cimg_op + intent_emb`（BEV 模式）<br>`zpo + zsp + zpa + zsa + pb + cim + cop + intent_emb`（FPV 模式） |

### 损失函数变量

| 变量名 | 全称 | 含义 | 计算方式 |
|--------|------|------|---------|
| **ploss** | **P**osition loss | 位置轨迹 LSTMVAE 的重构损失 + KL 散度 | `pos_encoder(x) → loss` |
| **sloss** | **S**peed loss | 速度序列 LSTMVAE 的重构损失 + KL 散度 | `speed_encoder(x) → loss` |
| **pbloss** | **P**edestrian **B**ehavior loss | 行人行为序列 LSTMVAE 的重构 + KL 散度 | `ped_behavior_encoder(x) → loss` |
| **psloss** | **P**edestrian **S**cene loss | 场景属性序列 LSTMVAE 的重构 + KL 散度 | `scene_attribute_encoder(x) → loss` |
| **cofe_loss** | CoFE correction loss | CoFE 轨迹修正损失（预测位移与 GT 位移的 MSE，按时间步累积开方） | `sum(sqrt(MSE(corr_dec_out, offset_gt[t])))` |
| **mloss** | **M**ulti-task loss | 多任务聚合损失（VAE 总损失 + CoFE 损失） | `ploss + sloss + pbloss + psloss + cofe_loss_weight * cofe_loss` |
| **speed_loss** | Speed prediction loss | 未来速度预测 MSE 损失（训练时除以 100 缩放） | `MSE(speed_preds, future_speed) / 100` |
| **total_loss** | 总训练损失 | 最终用于反向传播的标量损失 | `speed_loss + mloss` |

#### VAE 损失分解

每个 LSTMVAE 的 `loss_function` 返回两个子损失：

| 子损失 | 含义 | 公式 |
|--------|------|------|
| **Reconstruction_Loss** | 重构损失 | `MSE(x_recon, x_input)` |
| **KLD** | KL 散度（取负后为正） | `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`，带有 `kld_weight = 0.00025` |

### 模型输出

| 输出 | 含义 | 形状 | 说明 |
|------|------|------|------|
| `mloss` | 多任务聚合损失（标量） | `(1,)` | 训练时用于梯度计算 |
| `cofe_loss` | CoFE 修正损失（标量） | `(1,)` | 仅在训练模式 > 0，评估模式为 0 |
| `speed_preds` | 未来速度预测序列 | `(batch, output_len, 2)` | LSTMCell 自回归解码输出 |
| `speed_outputs` | 经 HardTanh 裁剪的速度预测 | `(batch, output_len, 2)` | 范围在 `[-hardtanh_limit, hardtanh_limit]` |

---

## 配置文件参数详解

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `dataset` | 数据集名称（`jaad`、`pie`、`titan`、`T2FPV`） | `T2FPV` |
| `batch_size` | 批次大小 | 8 |
| `n_epochs` | 训练轮数 | 50 |
| `lr` | 学习率 | 0.001 |
| `hidden_size` | LSTM 隐藏层维度（所有编码器统一） | 512 |
| `hardtanh_limit` | HardTanh 激活函数裁剪边界 | 100 |
| `use_image` | 是否使用图像特征 | `True` |
| `image_network` | 图像主干网络（`clstm`、`resnet50`、`resnet18`） | `clstm` |
| `use_opticalflow` | 是否使用光流特征 | `False` |
| `use_attribute` | 是否使用属性特征（行为、场景、行人属性） | `False` |
| `intent_feature_dim` | VLM 意图特征维度（0 表示不使用），如 CLIP 文本嵌入为 512 | 0 |
| `input` | 历史输入帧数 | 5 |
| `output` | 未来预测帧数 | 5 |
| `stride` | 滑动窗口采样步长 | 5 |
| `skip` | 帧间隔（每 skip 帧取一帧） | 1 |
| `lr_scheduler` | 是否使用 StepLR 学习率调度器 | `False` |
| `loader_workers` | DataLoader 工作线程数 | 8 |
| `loader_shuffle` | 是否打乱训练数据 | `True` |
| `pin_memory` | 是否使用 CUDA pin_memory 加速 | `True` |
| `prefetch_factor` | DataLoader 预取因子 | 3 |

### CoFE 模块参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `use_cofe` | 是否启用 CoFE 轨迹去噪 | `True` |
| `cofe_hidden_size` | CoFE GRU 隐藏层维度 | 96 |
| `cofe_num_layers` | CoFE GRU 层数 | 2 |
| `cofe_use_resnet` | CoFE 是否使用 ResNet 视觉特征（仅 FPV 模式） | `False` |
| `cofe_loss_weight` | CoFE 损失权重（>0 时 CoFE 参与端到端训练） | `1.0` |

### LSTMVAE 内部参数（vae.py）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `latent_size` | VAE 隐变量维度（等于 `hidden_size`） | 512 |
| `num_layers` | LSTM 层数（固定为 1） | 1 |
| `kld_weight` | KL 散度损失权重 | 0.00025 |

### T2FPV 数据集参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `hist_len` | 预处理时的历史帧数（预处理参数，不等于 `input`） | 8 |
| `fut_len` | 预处理时的未来帧数 | 12 |
| `frame_skip` | 帧采样间隔 | 10 |
| `min_agents` | 每个场景最小行人数 | 2 |

### 命名约定速查表

| 缩写 | 全称 | 含义 |
|------|------|------|
| **h** | Hidden | LSTM 隐藏状态 |
| **z** | Latent Z | VAE 隐变量 |
| **c** | Cell | LSTM 细胞状态 |
| **p** | Position | 位置轨迹 |
| **s** | Speed | 速度序列 |
| **pa** | Pedestrian Attribute | 行人行为/属性 |
| **sa** | Scene Attribute | 场景属性 |
| **b** | Behavior / Embedding | 行为或嵌入向量 |
| **img** | Image | 图像特征 |
| **op** | Optical flow | 光流特征 |
| **ds** | Decoder Sum | 解码器融合状态 |
| **cofe** | Correction Feature Embedding | 轨迹修正模块 |

---

## 数据流详解

### FPV 模式（T2FPV 数据集）完整前向路径

```
DataLoader batch:
  ├── pos:        (B, T_in, 2)        — 历史世界坐标
  ├── speed:      (B, T_in, 2)        — 历史速度
  ├── future_pos: (B, T_out, 2)       — 未来世界坐标（GT）
  ├── future_speed: (B, T_out, 2)     — 未来速度（GT）
  ├── hist_all:   (B, T, N, 7)        — 场景全部agent历史 [x,y,yaw,img_x,img_y,valid,id]
  ├── hist_yaw:   (B, T, N)           — 场景全部agent偏航角
  ├── seq_start_end: (B, 2)           — 场景边界索引
  ├── ego_idx:    (B,)                — 当前agent在场景中的索引
  └── intent_feature: (B, D) 可选     — VLM 意图特征

网络内部流程:
  1. hist_all → _normalize_fpv_inputs → (T, B*N, F)
  2. [train] CoFE.train_correction(gt_abs, pred_abs, intent) → cofe_loss
  3. CoFE.infer_correction(pred_abs, intent) → corrected_abs (T, B*N, 2)
  4. corrected_abs → permute → pos (B*N, T, 2)
  5. pos → 一阶差分 → speed (B*N, T, 2)
  6. speed + pos → LSTMVAE编码器 → hpo, zpo, hsp, zsp, ploss, sloss
  7. 特征融合: hds = hpo + hsp + ... + intent_emb
  8. LSTMCell 自回归解码 → speed_preds (B*N, T_out, 2)
  9. loss = speed_loss + mloss

训练监督信号:
  - cofe_loss:  CoFE预测位移 vs GT位移  (训练模式)
  - ploss/sloss: VAE重构 vs 输入序列     (始终)
  - speed_loss: 预测速度 vs future_speed  (始终)
```

### BEV 模式（JAAD/PIE/TITAN）

```
DataLoader batch:
  ├── pos:             (B, T_in, 4)    — 历史边界框 [x, y, w, h]
  ├── speed:           (B, T_in, 4)    — 历史边界框速度
  ├── future_pos:      (B, T_out, 4)   — 未来边界框（GT）
  ├── future_speed:    (B, T_out, 4)   — 未来速度（GT）
  ├── images:          (B, T, C, H, W) — 场景图像序列
  ├── optical:         (B, T, 4, H, W) — 光流序列
  ├── ped_behavior:    (B, T, N_beh)   — 行人行为标签
  ├── ped_attribute:   (B, N_attr)     — 行人属性
  └── scene_attribute: (B, N_scene)    — 场景属性

  流程:
  1. [CoFE] pos → CoFE修正 → corrected_pos
  2. corrected_pos → 重算速度
  3. 各模态独立编码 (LSTMVAE) → hpo, zpo, hsp, zsp, ...
  4. 特征融合 → hds, zds
  5. LSTMCell 解码 → speed_preds
```

---

## 支持的数据集

### BEV 视角数据集

| 数据集 | 说明 | 特点 |
|-------|------|------|
| JAAD | Joint Attention in Autonomous Driving | 包含丰富的行人行为标注 |
| PIE | Pedestrian Intention Estimation | 专注于意图预测 |
| TITAN | Traffic Intelligence in Autonomous Navigation | 自动驾驶场景 |

### FPV 视角数据集

| 数据集 | 说明 | 特点 |
|-------|------|------|
| **T2FPV** | Trajectory to First-Person View | 第一人称视角轨迹预测，包含检测噪声和修正 |

#### T2FPV 数据集支持

T2FPV 数据集特点：

- **第一人称视角**：车载相机拍摄的行人轨迹
- **检测噪声**：包含检测错误、ID跳跃、遮挡等问题
- **CoFE修正**：专门设计用于修正此类检测噪声
- **世界坐标**：轨迹以世界坐标 (x, y) 存储

数据格式：

```python
hist_all: (T, N, 7)
  [x_world, y_world, yaw, img_x, img_y, valid, agent_id]
```

---

## 安装

### 环境要求

- **操作系统**：Ubuntu 20.04 或更高版本
- **Python**：3.8 或更高版本
- **CUDA**：11.1 或更高版本（GPU 支持）
- **PyTorch**：支持 CUDA 的版本

### 安装步骤

```bash
git clone https://github.com/oAyanameio/OurWork1.git
cd OurWork1
python3 -m venv ptinet_env
source ptinet_env/bin/activate
pip install -r requirements.txt
```

> ⚠️ 如果使用 GPU，请确保安装支持 CUDA 的 PyTorch。

验证 CUDA 支持：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 数据预处理

### BEV 数据集（JAAD、PIE、TITAN）

#### 第一步：下载数据集

- [JAAD](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/)
- [PIE](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)
- [TITAN](https://usa.honda-ri.com/titan)

#### 第二步：下载 RAFT 并计算光流

```bash
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
pip install -r requirements.txt
```

按照 RAFT 文档为数据集计算稠密光流。

#### 第三步：组织数据

```
OurWork1/
└── data/
    ├── JAAD/
    │   ├── images/
    │   └── optical_flow/
    ├── PIE/
    │   ├── images/
    │   └── optical_flow/
    └── TITAN/
```

#### 第四步：预处理

```bash
python preprocess_data.py --dataset jaad
python preprocess_data.py --dataset pie
python preprocess_data.py --dataset titan
```

处理后的文件将保存在 `processed/` 目录下。

### FPV 数据集（T2FPV）

T2FPV 数据集预处理：

```bash
# 1. 下载原始数据集
# 链接：https://cmu.box.com/s/tij0yyo8ulqh1n7uane0pf3onj7ror7f

# 2. 解压数据
mkdir -p data/
tar -xvf FPVDataset.tar.gz -C data/
mv data/FPVDataset data/T2FPV/

# 3. 数据目录结构
data/T2FPV/
├── imgs/              # 原始视频帧和ResNet特征
│   ├── biwi_eth/agent1/
│   └── ...
├── mp4/               # 视频文件
├── gt_dets/           # 真实检测标注
└── pred_dets/         # 预测检测（用于CoFE修正）
```

#### 运行预处理

```bash
python data/preprocess_t2fpv.py --data_root ./FPVDataset
```

预处理参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--hist_len` | 历史帧数 | 8 |
| `--fut_len` | 未来帧数 | 12 |
| `--frame_skip` | 帧采样间隔 | 10 |
| `--stride` | 滑动窗口步长 | 1 |
| `--min_agents` | 每个场景最小行人数 | 2 |
| `--folds` | 指定数据折（eth/hotel/univ/zara1/zara2） | 全部 |

---

## 训练

### 配置文件

模型配置通过 `config.yml` 管理：

```yaml
# 通用配置
dataset: 'T2FPV'
batch_size: 8
n_epochs: 50
lr: 0.001
hidden_size: 512

# 模态开关
use_image: True
image_network: 'clstm'
use_opticalflow: False
use_attribute: False

# VLM 意图特征
intent_feature_dim: 0  # 512 表示启用 CLIP 文本嵌入

# CoFE 配置
use_cofe: True
cofe_hidden_size: 96
cofe_num_layers: 2
cofe_loss_weight: 1.0  # CoFE 损失权重，控制 CoFE 参与训练的程度
```

### 启动训练

#### BEV 模式（JAAD/PIE/TITAN）

```bash
python train.py --dataset jaad
```

#### FPV 模式（T2FPV）

```bash
python train.py --dataset T2FPV
```

模型检查点保存在 `checkpoints/` 文件夹中。

---

## 测试与验证

### 运行测试套件

#### CoFE 模块优化测试

验证向量化优化的功能一致性和性能：

```bash
python test_cofe_optimization.py
```

**测试内容**：
- ✅ ego_dists 函数功能一致性（数值误差 < 1e-8）
- ✅ encode_yaw 函数功能一致性
- ✅ 边界条件测试（单个场景、单个agent）
- ✅ 性能对比测试（加速比）
- ✅ 完整 CoFE pipeline 测试

#### PTINet 集成测试

验证 CoFE 与 PTINet 的集成是否正常工作：

```bash
python test_ptinet_integration.py
```

**测试内容**：
- ✅ PTINet + CoFE 初始化
- ✅ FPV 模式前向传播
- ✅ 输出形状验证

---

## 项目结构

```
OurWork1/
├── model/
│   ├── network_image.py         # 主模型定义（PTINet + CoFE 集成）
│   ├── cofe.py                 # CoFE 轨迹去噪修复模块（向量化优化）
│   ├── vae.py                   # LSTMVAE 变分自编码器
│   └── clstm.py                 # ConvLSTM 时空卷积模块
├── datasets/
│   ├── __init__.py              # 数据集注册入口
│   ├── t2fpv.py                 # T2FPV 数据集 PyTorch 注册
│   ├── jaad.py                  # JAAD 数据集加载器
│   ├── pie.py                   # PIE 数据集加载器
│   └── titan.py                 # TITAN 数据集加载器
├── data/
│   ├── t2fpv_dataset.py         # T2FPV 数据集 PyTorch Dataset 类
│   ├── preprocess_t2fpv.py     # T2FPV 预处理脚本
│   ├── processed/               # 预处理输出目录
│   └── ...
├── preprocess/
│   ├── jaad_preprocessor.py     # JAAD 数据预处理
│   ├── pie_preprocessor.py      # PIE 数据预处理
│   └── ...
├── visualization/
│   ├── visualize.py             # 可视化工具
│   └── display.py               # 显示函数
├── train.py                     # 分布式训练入口
├── utils.py                     # 工具函数（ADE/FDE/速度转位置等）
├── config.yml                   # 配置文件
├── f1_score.py                  # F1 评分计算
├── test_cofe_optimization.py    # CoFE 优化测试套件
├── test_ptinet_integration.py   # PTINet 集成测试
├── requirements.txt             # 依赖清单
└── 参考文献/                     # 参考论文 PDF
```

---

## 学习资源

### 理解模型的关键概念

1. **LSTMVAE**：核心序列建模组件。每个模态（位置、速度、行为等）都有一个独立的 LSTMVAE，将输入序列编码到隐空间，再重构回输入空间。VAE 损失由重构损失（MSE）和 KL 散度两部分组成。

2. **特征融合策略**：所有编码器的输出通过简单的加法融合（`hds = hpo + hsp + ...`），这种方法轻量高效，但要求所有隐藏状态维度一致（统一为 `hidden_size`）。

3. **自回归解码**：使用 LSTMCell 逐时间步预测未来。当前步的输入来自上一步的预测输出（经过 `detach()` 切断梯度），每个时间步更新 `(hds, zds)` 状态。

4. **CoFE 训练模式**：仅在 `self.training=True` 时计算 `cofe_loss`。评估/推理时只运行 `infer_correction` 进行轨迹修正，不计算损失。

5. **物理一致性**：CoFE 修正后的位置通过一阶差分计算速度，确保位置和速度之间的物理约束始终成立，避免模型学到不一致的轨迹-速度映射。

6. **VLM 意图锚点**：当 `intent_feature_dim > 0` 时，VLM 提取的高级语义特征通过 MLP 投影到 `hidden_size` 维度，在 CoFE 编解码和最终解码器融合中均参与计算。

---

## 参考文献

本模型基于以下工作扩展：

```bibtex
@article{munir2024context,
  title        = {Context-aware multi-task learning for pedestrian intent and trajectory prediction},
  author       = {Munir, Farzeen and Kucner, Tomasz},
  journal      = {Transportation Research Part C: Emerging Technologies},
  volume       = {160},
  pages        = {104762},
  year         = {2024},
  publisher    = {Elsevier}
}

@article{stoler2023t2fpv,
  title        = {T2FPV: Dataset and Method for Correcting First-Person View Errors in Pedestrian Trajectory Prediction},
  author       = {Stoler, Ben and Sathyanarayana, Darshan and Kucner, Tomasz and Lind, Franz and Seg, Achim},
  year         = {2023}
}
```

CoFE 模块的设计参考了 T2FPV 项目中的轨迹修正方法。

---

## 联系方式

如有问题，请提交 Issue 或通过 GitHub 联系。

---

## 更新日志

### 2024-05-20

- ✅ 集成 VLM 语义意图特征作为 CoFE 的"语义锚点"
- ✅ 修复 CoFE 梯度回传 Bug（打通 GT 信号传递 + 激活 cofe_loss_weight）
- ✅ 添加 hpo/zpo/hsp/zsp 等模型内部变量详细说明文档
- ✅ 完善 T2FPV FPV 数据流文档
- ✅ 新增 VAE 损失分解说明

### 2024-05-18

- ✅ 完成 CoFE 模块向量化优化
- ✅ 添加 CoFE 模块功能测试套件
- ✅ 添加 PTINet 集成测试
- ✅ 完善 T2FPV 数据集支持文档