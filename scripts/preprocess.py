"""
T2FPV 数据预处理入口脚本

用法:
    python scripts/preprocess.py --data_root /path/to/FPVDataset
    python scripts/preprocess.py --folds eth hotel
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.preprocess import main

if __name__ == '__main__':
    main()