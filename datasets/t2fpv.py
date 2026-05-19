"""
T2FPV 数据集注册模块

将 data/t2fpv_dataset 中的 T2FPV 数据集类注册到 datasets 包中，
使 train.py 可通过 eval('datasets.T2FPV') 调用。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
from t2fpv_dataset import T2FPV
