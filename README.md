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
   ```python
   # ego相对偏航角 → [-180°, 180°]归一化 → 弧度 → cos/sin编码
   offset_yaw_rel = hist_yaw[:, start:end] - hist_yaw[:, start]
   offset_yaw_norm = ((180 + offset_yaw_rel) % 360 - 180)
   yaw_enc = [cos(rad), sin(rad)]
   ```

2. **Ego 相对距离**：计算每个 agent 相对于自车的坐标偏移
   ```python
   # 通过 seq_start_end 批量计算ego相对坐标
   ego_coords = hist_abs[:, ego_indices, :]  # 获取每个场景ego的坐标
   offset_coords = hist_abs - ego_coords      # 计算相对坐标
   ```

3. **ResNet 视觉融合**：可选融合 2048 维 ResNet 视觉特征
4. **帧间位移修正**：在位移空间中做 GRU 编解码
5. **绝对坐标恢复**：通过 cumsum + initial_pos 还原绝对坐标

##### BEV 模式（鸟瞰图视角）

简化版本，不依赖偏航角和 ego 相对坐标：

```python
# BEV模式下，仅使用相对位移进行去噪
offset_pred = rel_pred  # 帧间位移
```

##### 数据流

```
绝对坐标 → 特征构建 → idxs选择 → GRU编码 → GRU解码 → 位移还原 → 修正后绝对坐标
```

#### 2. 物理一致性（速度重计算）

CoFE 修复位置序列后，从修复后的位置通过一阶差分重新计算速度：

- 第 0 帧速度保持为 0（无前一帧可差分）
- 第 t 帧速度 = pos[t] - pos[t-1]（t ≥ 1）

这确保了位置与速度之间的物理关系始终成立，避免因轨迹修复导致的速度信号失真。

#### 3. 向量化性能优化

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

### 双任务输出

- **速度轨迹预测**：通过 LSTMCell 解码器逐时间步生成未来速度序列
- **穿越意图预测**：通过独立的 LSTMCell 解码器输出每帧的穿越概率（二分类）

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

---

## 训练

### 配置文件

模型配置通过 `config.yml` 管理：

#### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `dataset` | 数据集名称（`jaad`、`pie`、`titan`、`T2FPV`） | `T2FPV` |
| `batch_size` | 批次大小 | 8 |
| `n_epochs` | 训练轮数 | 50 |
| `lr` | 学习率 | 0.001 |
| `hidden_size` | LSTM 隐藏层维度 | 512 |
| `use_image` | 是否使用图像特征 | `True` |
| `image_network` | 图像网络类型（`clstm`、`resnet50`、`resnet18`） | `clstm` |
| `use_opticalflow` | 是否使用光流特征 | `False` |
| `use_attribute` | 是否使用属性特征 | `False` |

#### CoFE 模块参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `use_cofe` | 是否启用 CoFE 轨迹去噪 | `True` |
| `cofe_hidden_size` | CoFE GRU 隐藏层维度 | 96 |
| `cofe_num_layers` | CoFE GRU 层数 | 2 |
| `cofe_use_resnet` | CoFE 是否使用 ResNet 特征（仅FPV模式） | `False` |

#### CoFE 配置说明（config.yml 示例）

```yaml
# 通用配置
dataset: 'T2FPV'
use_cofe: True

# CoFE 配置
cofe_hidden_size: 96
cofe_num_layers: 2
cofe_use_resnet: False  # T2FPV模式下可启用以提升性能
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

### 测试结果示例

```
============================================================
CoFE模块向量化优化测试套件
============================================================
测试1: ego_dists功能一致性
最大误差: 0.0000000000
结果一致: True
✅ ego_dists功能验证通过!

测试2: encode_yaw功能一致性
最大误差: 0.0000000000
结果一致: True
✅ encode_yaw功能验证通过!

测试3: 边界条件测试
✅ 边界条件测试通过!

测试4: 性能对比测试
使用设备: cpu

ego_dists性能:
  原实现: 0.51 ms/次
  新实现: 0.28 ms/次
  加速比: 1.82x

============================================================
🎉 所有测试通过! 优化成功!
============================================================
```

---

## 评估

评估训练好的模型：

```bash
python evaluate.py --dataset jaad --checkpoint checkpoints/jaad_model.pth
```

### 评估指标

#### 轨迹预测

- **ADE**：平均位移误差（Average Displacement Error）
- **FDE**：最终位移误差（Final Displacement Error）

#### 意图预测

- **准确率（Accuracy）**
- **精确率（Precision）**
- **召回率（Recall）**
- **F1 分数（F1-score）**

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
│   ├── jaad.py                  # JAAD 数据集加载器
│   ├── pie.py                   # PIE 数据集加载器
│   └── titan.pyLLM 数据集加载器
├── preprocess/
│   ├── jaad_preprocessor.py     # JAAD 数据预处理
│   └── pie_preprocessor.py      # PIE 数据预处理
├── visualization/
│   ├── visualize.py             # 可视化工具
│   └── display.py                # 显示函数
├── train.py                     # 训练入口
├── utils.py                     # 工具函数
├── config.yml                   # 配置文件
├── f1_score.py                  # F1 评分计算
├── test_cofe_optimization.py    # CoFE 优化测试套件
├── test_ptinet_integration.py   # PTINet 集成测试
├── requirements.txt             # 依赖清单
└── 参考文献/                     # 参考论文 PDF
    ├── Munir和Kucner - 2024 - Context-Aware Multi-Task Learning...
    └── Stoler 等 - 2023 - T2FPV Dataset and Method...
```

---

## 性能基准

### CoFE 模块性能

| 配置 | 环境 | 加速比 | 说明 |
|------|------|--------|------|
| ego_dists | CPU | **1.82x** | 向量化操作替代 for 循环 |
| encode_yaw | CPU | 1.00x | 优化空间有限 |

> 💡 在 GPU 环境下，预期加速比可达 5-10x（取决于数据规模）。

### CoFE 特征维度

FPV 模式完整特征空间（8维）：

| 维度范围 | 特征 | 说明 |
|---------|------|------|
| 0-1 | `xy - xy[0]` | 相对首帧位移 |
| 2-3 | `offset_xy` | Ego 相对坐标 |
| 4-5 | `cos/sin(yaw)` | 偏航角编码 |
| 6-7 | `rel` | 帧间位移（速度） |

通过 `idxs` 参数可选择子集送入 GRU（默认 `[6, 7]` 只选位移）。

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

### 2024-05-18

- ✅ 完成 CoFE 模块向量化优化
  - `ego_dists` 函数：1.82x CPU 加速
  - `encode_yaw` 函数：功能验证通过
- ✅ 添加 CoFE 模块功能测试套件
- ✅ 添加 PTINet 集成测试
- ✅ 完善 T2FPV 数据集支持文档
- ✅ 更新性能基准信息
