"""
PTINet 训练入口脚本

用法:
    python scripts/train.py
    配置从 config/default.yml 读取
"""

import os
import sys
import datetime
import argparse
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import datasets
from training.trainer import train


def parse_config_file(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Train PTINet network')

    parser.add_argument('--data_dir', type=str,
                        default='/home/farzeen/work/aa_postdoc/intent/JAAD/PN/',
                        required=False, help='数据集目录路径')
    parser.add_argument('--dataset', type=str,
                        default='pie',
                        required=False, help='数据集名称（jaad/pie）')
    parser.add_argument('--out_dir', type=str,
                        default='/home/farzeen/work/aa_postdoc/intent/PIE_bbox_image/bounding-box-prediction/output',
                        required=False, help='输出目录路径')

    parser.add_argument('--input', type=int,
                        default=16,
                        required=False, help='输入序列长度（历史帧数）')
    parser.add_argument('--output', type=int,
                        default=32,
                        required=False, help='输出序列长度（预测帧数）')
    parser.add_argument('--stride', type=int,
                        default=16,
                        required=False, help='采样步长')
    parser.add_argument('--skip', type=int, default=1, help='帧间隔')

    parser.add_argument('--dtype', type=str, default='train', help='数据类型')
    parser.add_argument("--from_file", type=bool, default=False, help='是否从文件加载预处理数据')
    parser.add_argument('--save', type=bool, default=True, help='是否保存训练结果')
    parser.add_argument('--log_name', type=str, default='', help='日志文件夹名称')
    parser.add_argument('--loader_workers', type=int, default=16, help='DataLoader 工作线程数')
    parser.add_argument('--loader_shuffle', type=bool, default=True, help='是否打乱数据顺序')
    parser.add_argument('--pin_memory', type=bool, default=False, help='是否使用 pin_memory')
    parser.add_argument('--prefetch_factor', type=int, default=3, help='预取因子')

    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--batch_size', type=int, default=4, help='单卡批次大小')
    parser.add_argument('--n_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--lr_scheduler', type=bool, default=False, help='是否使用学习率调度器')
    parser.add_argument('--local-rank', type=int, default=0, help='本地进程排名')

    parser.add_argument('--hidden_size', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--hardtanh_limit', type=int, default=100, help='HardTanh 激活函数边界')
    parser.add_argument('--use_image', type=bool, default=False,
                        help='使用图像作为输入特征')
    parser.add_argument('--image_network', type=str, default='clstm',
                        help='选择图像主干网络（clstm）')
    parser.add_argument('--use_attribute', type=bool, default=True,
                        help='使用属性作为输入特征')
    parser.add_argument('--use_opticalflow', type=bool, default=True,
                        help='使用光流作为输入特征')

    parser.add_argument('--cofe_frozen', type=bool, default=False,
                        help='是否冻结 CoFE 模块（第二阶段训练时设为 True）')
    parser.add_argument('--cofe_pretrained', type=str, default='',
                        help='预训练 CoFE 权重路径（第二阶段训练时加载）')

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
    train(args, train_set, val_set)