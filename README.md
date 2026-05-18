# OurWork1：基于 CoFE 去噪增强的行人轨迹与意图预测网络

OurWork1 是在 PTINet 基础上扩展的多任务学习框架，用于联合预测行人轨迹与穿越意图。相比于原始 PTINet，本项目引入了 **CoFE（Correction Feature Embedding）轨迹去噪修复模块**，在历史轨迹送入 LSTM-VAE 编码之前先进行去噪修复，并确保修复后位置与速度的物理一致性。

该框架适用于自动驾驶和机器人导航等场景——在这些场景中，预判行人运动对于安全性和效率至关重要。

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

CoFE 基于 GRU 编码器-解码器架构，输入为带噪声的历史轨迹 \([seq\_len, batch\_size, 4]\)，输出为修复后的轨迹（形状相同）。其工作原理：

1. **编码阶段**：逐时间步将带噪声轨迹编码到隐空间
2. **解码阶段**：从隐状态逐时间步解码出修正后的轨迹

该模块去除了 T2FPV 原始 CoFE 实现中依赖 FPV 视角的 yaw 角度编码、ResNet 视觉融合、ego 相对距离等组件，保留了核心的序列编解码能力，适配 BEV 视角的轨迹预测任务。

#### 2. 物理一致性（速度重计算）

CoFE 细化）

CoFE 修复位置序列后，从修复后的位置通过一阶差分重新计算速度：

- 第 0 帧速度保持为 0（无前一帧可差分）
- 第 t 帧速度 = pos[t] - pos[t-1]（t ≥ 1）

这确保了位置与速度之间的物理关系始终成立，避免因轨迹修复导致的速度信号失真。

### 多模态输入

| 输入模态 | 说明 | 编码方式 |
|---------|------|---------|
| **轨迹位置** | 历史边界框序列 (x, y, w, h) | LSTMVAE 编码器 |
| **轨迹速度** | 帧间差分速度序列 (x, y, w, h) | LSTMVAE 编码器 |
| **行人行为** | 反应、手势、注视、点头等行为标签 | LSTMVAE 编码器 |
| **场景属性** | 交通标志、道路类型等场景特征 | LSTMVAE 编码器 |
| **行人属性** | 年龄、性别、群体大小 | MLP 编码器 |
| **视觉图像** | 场景 RGB 图像序列 | ConvLSTM / ResNet50 / ResNet18 |
| **光流** | 相邻帧稠密光流 | ResNet50 + LSTM |

### 双任务输出

- **速度轨迹预测**：通过 LSTMCell 解码器逐时间步生成未来速度序列
- **穿越意图预测**：通过独立的 LSTMCell 解码器输出每帧的穿越概率（二分类）

---

## 📦 安装

### 环境要求

- **操作系统**：Ubuntu 20.04 或更高版本
- **Python**：3.8 或更高版本
- **CUDA**：11.1 或更高版本（GPU 支持）

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

## 🧹 数据预处理

OurWork1 支持 JAAD、PIE 和 TITAN 数据集。需要预先计算稠密光流作为输入。

### 第一步：下载数据集

- [JAAD](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/)
- [PIE](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)
- [TITAN](https://usa.honda-ri.com/titan)

### 第二步：下载 RAFT 并计算光流

```bash
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
pip install -r requirements.txt
```

按照 RAFT 文档为数据集计算稠密光流。

### 第三步：组织数据

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

### 第四步：预处理

```bash
python preprocess_data.py --dataset jaad
python preprocess_data.py --dataset pie
python preprocess_data.py --dataset titan
```

处理后的文件将保存在 `processed/` 目录下。

---

## 🏋️‍♂️ 训练

### 配置文件

模型配置通过 `config.yml` 管理，关键参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `dataset` | 数据集名称（`jaad`、`pie`、`titan`） | `jaad` |
| `batch_size` | 批次大小 | 8 |
| `n_epochs` | 训练轮数 | 50 |
| `lr` | 学习率 | 0.001 |
| `hidden_size` | LSTM 隐藏层维度 | 512 |
| `use_cofe` | 是否启用 CoFE 轨迹去噪 | `True` |
| `cofe_hidden_size` | CoFE GRU 隐藏层维度 | 96 |
| `cofe_num_layers` | CoFE GRU 层数 | 2 |
| `use_image` | 是否使用图像特征 | `True` |
| `image_network` | 图像网络类型（`clstm`、`resnet50`、`resnet18`） | `clstm` |
| `use_opticalflow` | 是否使用光流特征 | `True` |
| `use_attribute` | 是否使用属性特征 | `True` |

### 启动训练

```bash
python train.py
```

模型检查点保存在 `checkpoints/` 文件夹中。

---

## 📊 评估

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

## 🧩 项目结构

```
OurWork1/
├── model/
│   ├── network_image.py    # 主模型定义（PTINet + CoFE 集成）
│   ├── cofe.py             # CoFE 轨迹去噪修复模块
│   ├── vae.py              # LSTMVAE 变分自编码器
│   └── clstm.py            # ConvLSTM 时空卷积模块
├── datasets/
│   ├── jaad.py             # JAAD 数据集加载器
│   ├── pie.py              # PIE 数据集加载器
│   └── titan.py            # TITAN 数据集加载器
├── preprocess/
│   ├── jaad_preprocessor.py # JAAD 数据预处理
│   └── pie_preprocessor.py  # PIE 数据预处理
├── visualization/
│   ├── visualize.py         └── display.py           # 显示函数
├── train.py                 # 训练入口
├── utils.py                 # 工具函数
├── config.yml               # 配置文件
├── f1_score.py              # F1 评分计算
└── requirements.txt         # 依赖清单
```

---

## 📚 参考文献

本模型基于以下工作扩展：

```bibtex
@article{munir2025context,
  title        = {Context-aware multi-task learning for pedestrian intent and trajectory prediction},
  author       = {Munir, Farzeen},
  journal      = {Transportation Research Part C},
  year         = {2025},
  publisher    = {Elsevier},
}
```

CoFE 模块的设计参考了 T2FPV 项目中的轨迹修正方法。

---

## 📫 联系方式

如有问题，请提交 Issue 或通过 GitHub 联系。