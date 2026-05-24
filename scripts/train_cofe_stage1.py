"""
CoFE 第一阶段预训练入口脚本

仅训练 CoFE 模块的轨迹去噪能力，不训练主网络。
训练完成后保存 CoFE 权重供第二阶段加载。

用法:
    python scripts/train_cofe_stage1.py
    配置从 config/default.yml 读取
"""

import os
import sys
import argparse
import datetime
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import datasets
from training.cofe_stage1 import train_cofe_stage1


def parse_config_file(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='CoFE Stage 1 Pre-training')

    parser.add_argument('--data_dir', type=str,
                        default='/home/lbh/PTINet/data/JAAD/',
                        help='数据集目录路径')
    parser.add_argument('--dataset', type=str,
                        default='T2FPV',
                        help='数据集名称')
    parser.add_argument('--out_dir', type=str,
                        default='/home/lbh/OurWork1/output/',
                        help='输出目录路径')

    parser.add_argument('--input', type=int, default=5, help='输入序列长度')
    parser.add_argument('--output', type=int, default=5, help='输出序列长度')
    parser.add_argument('--stride', type=int, default=5, help='采样步长')
    parser.add_argument('--skip', type=int, default=1, help='帧间隔')

    parser.add_argument('--dtype', type=str, default='train', help='数据类型')
    parser.add_argument('--from_file', type=bool, default=False, help='是否从文件加载')
    parser.add_argument('--save', type=bool, default=True, help='是否保存')
    parser.add_argument('--log_name', type=str, default='cofe_stage1', help='日志文件夹名称')
    parser.add_argument('--loader_workers', type=int, default=8, help='DataLoader 工作线程数')
    parser.add_argument('--loader_shuffle', type=bool, default=True, help='是否打乱数据')
    parser.add_argument('--pin_memory', type=bool, default=True, help='是否使用 pin_memory')
    parser.add_argument('--prefetch_factor', type=int, default=3, help='预取因子')

    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--n_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--lr_scheduler', type=bool, default=False, help='是否使用学习率调度器')

    parser.add_argument('--hidden_size', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--hardtanh_limit', type=int, default=100, help='HardTanh 边界')

    parser.add_argument('--use_image', type=bool, default=False, help='Stage1 不需要图像')
    parser.add_argument('--image_network', type=str, default='clstm', help='图像网络类型')
    parser.add_argument('--use_attribute', type=bool, default=False, help='Stage1 不需要属性')
    parser.add_argument('--use_opticalflow', type=bool, default=False, help='Stage1 不需要光流')

    parser.add_argument('--cofe_save_name', type=str, default='cofe_stage1.pkl',
                        help='CoFE 权重保存文件名')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("Date and time:", datetime.datetime.now())
    args = parse_args()
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'default.yml')
    config = parse_config_file(config_path)
    if config.get('use_argument_parser') == False:
        for arg in vars(args):
            if arg in config:
                setattr(args, arg, config[arg])
    print(args)

    os.makedirs(os.path.join(args.out_dir, args.log_name), exist_ok=True)

    train_set = getattr(datasets, args.dataset)(
        data_dir=args.data_dir,
        out_dir=os.path.join(args.out_dir, args.log_name),
        dtype='train',
        input=args.input,
        output=args.output,
        stride=args.stride,
        skip=args.skip,
        from_file=args.from_file,
        save=args.save,
        use_images=args.use_image,
        use_attribute=args.use_attribute,
        use_opticalflow=args.use_opticalflow
    )
    val_set = getattr(datasets, args.dataset)(
        data_dir=args.data_dir,
        out_dir=os.path.join(args.out_dir, args.log_name),
        dtype='val',
        input=args.input,
        output=args.output,
        stride=args.stride,
        skip=args.skip,
        from_file=args.from_file,
        save=args.save,
        use_images=args.use_image,
        use_attribute=args.use_attribute,
        use_opticalflow=args.use_opticalflow
    )
    train_cofe_stage1(args, train_set, val_set)