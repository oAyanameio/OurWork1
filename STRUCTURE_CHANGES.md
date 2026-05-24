# 目录结构变更说明

## 概述

本文件记录了 OurWork1 (PTINet + CoFE 行人轨迹预测) 项目的目录结构优化调整。优化遵循软件工程最佳实践，以功能模块化为核心原则，提升项目的可维护性和可扩展性。

---

## 新旧结构对比

### 原结构

```
OurWork1/
├── model/                  # 模型定义（散放）
│   ├── network_image.py    # PTINet 主网络
│   ├── clstm.py            # ConvLSTM 特征提取
│   ├── vae.py              # LSTMVAE 编码器
│   └── cofe.py             # CoFE 去噪模块
├── datasets/               # 数据集注册模块
│   ├── __init__.py
│   └── t2fpv.py            # （含 sys.path hack）
├── data/                   # 数据加载与预处理
│   ├── t2fpv_dataset.py    # T2FPV Dataset 类
│   └── preprocess_t2fpv.py # 预处理脚本
├── train.py                # PTINet 训练（含完整逻辑）
├── train_cofe_stage1.py    # CoFE 预训练（含完整逻辑）
├── utils.py                # 工具函数（杂项）
├── f1_score.py             # 冗余文件
├── config.yml              # 配置文件（根目录）
├── 参考文献/               # 中文学术文献目录
├── test_ptinet_integration.py  # 测试文件（根目录）
├── test_cofe_optimization.py   # 测试文件（根目录）
└── preprocess/             # 预处理包（空壳）
```

### 新结构

```
OurWork1/
├── src/                    # 【新增】源代码根包
│   ├── __init__.py
│   ├── models/             # 模型定义模块
│   │   ├── __init__.py
│   │   ├── ptinet.py       # PTINet 主网络
│   │   ├── clstm.py        # ConvLSTM
│   │   ├── vae.py          # LSTMVAE
│   │   └── cofe.py         # CoFE
│   ├── data/               # 数据处理模块
│   │   ├── __init__.py
│   │   ├── t2fpv_dataset.py
│   │   └── preprocess.py   # 预处理（模块化，含 main()）
│   ├── datasets/           # 数据集注册（消除 sys.path hack）
│   │   └── __init__.py
│   ├── training/           # 训练逻辑模块
│   │   ├── __init__.py
│   │   ├── trainer.py      # PTINet 训练核心
│   │   └── cofe_stage1.py  # CoFE 预训练核心
│   └── utils/              # 工具函数模块
│       ├── __init__.py
│       └── metrics.py      # ADE/FDE/speed2pos 等指标
├── scripts/                # 【新增】CLI 入口脚本
│   ├── train.py            # PTINet 训练（薄入口）
│   ├── train_cofe_stage1.py# CoFE 预训练（薄入口）
│   ├── preprocess.py       # 数据预处理（包装调用）
│   ├── eval_cofe.py        # CoFE 独立评估
│   ├── extract_intent.py   # 意图特征提取（占位）
│   └── start_vllm.py       # VLLM 服务
├── config/                 # 【新增】配置文件目录
│   └── default.yml         # 默认配置
├── tests/                  # 【新增】测试目录
│   ├── __init__.py
│   ├── test_ptinet_integration.py
│   └── test_cofe_optimization.py
├── visualization/          # 可视化（保留）
│   ├── display.py
│   └── visualize.py
├── references/             # 【已重命名】学术参考文献
├── requirements.txt
├── .gitignore
└── STRUCTURE_CHANGES.md    # 【新增】本说明文件
```

---

## 主要调整说明

### 1. 源代码集中管理：新增 `src/` 根包

**问题**：原有代码散落在根目录的 `model/`、`data/`、`datasets/` 等多个目录，没有统一的代码组织层级。

**改进**：创建 `src/` 目录作为所有源代码的根包，按功能划分为 5 个子模块：
- `src/models/`：所有神经网络模型定义
- `src/data/`：数据集加载与预处理
- `src/datasets/`：数据集注册与动态发现
- `src/training/`：训练循环核心逻辑
- `src/utils/`：评估指标等工具函数

### 2. 逻辑与入口分离：新增 `scripts/` 目录

**问题**：`train.py` 和 `train_cofe_stage1.py` 将参数解析、数据加载、训练循环等全部逻辑混合在单个文件中，不利于复用和测试。

**改进**：采用"薄入口 + 厚逻辑"模式：
- `scripts/train.py`：仅包含参数解析和 CLI 入口，训练逻辑委托给 `src/training/trainer.py`
- `scripts/train_cofe_stage1.py`：同理，委托给 `src/training/cofe_stage1.py`
- `scripts/preprocess.py`：包装调用 `src/data/preprocess.main()`

新建脚本可通过在 `scripts/` 下添加入口文件快速扩展。

### 3. 消除 `sys.path.insert` hack

**问题**：原 `datasets/t2fpv.py` 通过 `sys.path.insert` 动态修改 Python 导入路径来导入 `data/` 下的模块，这是一种脆弱的做法。

**改进**：新结构采用统一的导入策略：
- CLI 入口脚本将 `src/` 加入 `sys.path`
- 所有 `src/` 下的模块使用标准导入（如 `from models.ptinet import PTINet`）
- `src/datasets/__init__.py` 直接导入 `data.t2fpv_dataset.T2FPV`，无路径 hack

### 4. 测试文件独立目录

**问题**：测试文件 `test_ptinet_integration.py` 和 `test_cofe_optimization.py` 散落在根目录。

**改进**：统一移至 `tests/` 目录，与源代码清晰分离。

### 5. 配置文件独立目录

**问题**：`config.yml` 位于根目录，与其他文件混杂。

**改进**：移至 `config/default.yml`，便于后续添加多套配置（如 `config/t2fpv.yml`、`config/jaad.yml`）。

### 6. 工具函数模块化

**问题**：`utils.py` 将所有工具函数混在一个文件中。

**改进**：移至 `src/utils/metrics.py`，形成 `src/utils/` 包，便于后续按功能添加更多工具模块（如 `src/utils/io.py`、`src/utils/visualization.py`）。

### 7. 中文学术文献目录重命名

**问题**：`参考文献/` 使用中文命名，在命令行操作和跨平台兼容性上存在不便。

**改进**：重命名为 `references/`，保持英文命名规范。

---

## 导入路径对照

| 原路径 | 新路径 |
|---|---|
| `import model.network_image as network` | 脚本中：`sys.path.insert + from models import ptinet as network` |
| `from model.clstm import ConvLSTM` | `from models.clstm import ConvLSTM` |
| `from data.t2fpv_dataset import T2FPV` | 同上（已在正确的 sys.path 上下文） |
| `import utils; utils.ADE(...)` | `from utils.metrics import ADE` |

---

## 运行方式

```bash
# PTINet 训练
python scripts/train.py --dataset T2FPV --device cuda

# CoFE 第一阶段预训练
python scripts/train_cofe_stage1.py --dataset T2FPV --device cuda

# 数据预处理
python scripts/preprocess.py --data_root /path/to/FPVDataset

# 运行测试
python tests/test_ptinet_integration.py
python tests/test_cofe_optimization.py
```

> **注意**：原有的根目录文件（`train.py`、`utils.py` 等）均保留，但推荐使用新的 `scripts/` 和 `src/` 结构开发新功能。