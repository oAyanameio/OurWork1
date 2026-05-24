"""
T2FPV 数据集注册模块

将 src.data.t2fpv_dataset 中的 T2FPV 数据集类注册到 datasets 包中，
使 scripts/train.py 可通过 getattr(datasets, 'T2FPV') 调用。
"""

from data.t2fpv_dataset import T2FPV