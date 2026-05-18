"""
PTINet 分布式训练脚本

该脚本用于训练行人意图预测网络（PTINet），支持多GPU分布式训练模式。
主要功能包括：
- 使用 PyTorch DistributedDataParallel 进行多卡训练
- 支持分布式数据采样（DistributedSampler）
- 解析配置文件和命令行参数
- 加载 JAAD/PIE 数据集
- 训练网络并进行验证
- 保存最佳模型和训练日志

依赖模块：
- torch: 深度学习框架（含分布式训练支持）
- torch.distributed: 分布式训练模块
- numpy/pandas: 数据处理
- sklearn: 指标计算
- datasets: 自定义数据集模块
- model.network_image: PTINet 模型定义
- utils: 工具函数（指标计算、坐标转换等）
- visualization.display: 可视化工具（可选）
"""

import time
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 分布式训练相关导入
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, precision_score, f1_score
import datetime
import datasets

# 模型导入
import model.network_image as network

# 工具函数导入
import utils
from utils import data_loader, calculate_score
from torch.utils.tensorboard import SummaryWriter

# 可视化模块（可选使用）
import visualization.display as viz

def parse_config_file(file_path):
    """
    解析 YAML 配置文件
    
    Args:
        file_path (str): 配置文件的路径
        
    Returns:
        dict: 配置参数字典
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
    

def parse_args():
    """
    解析命令行参数（分布式训练版本）
    
    Returns:
        argparse.Namespace: 包含所有参数的命名空间对象
        
    参数说明:
        --data_dir: 数据集目录路径
        --dataset: 数据集名称，支持 'jaad' 或 'pie'，默认 'pie'
        --out_dir: 输出目录路径
        --input: 输入序列长度（历史帧数），默认 16
        --output: 输出序列长度（预测帧数），默认 32
        --stride: 采样步长，默认 16
        --skip: 帧间隔，默认 1
        --dtype: 数据类型（train/val/test），默认 'train'
        --from_file: 是否从文件加载预处理数据，默认 False
        --save: 是否保存训练结果，默认 True
        --log_name: 日志文件夹名称，默认为空（自动生成）
        --loader_workers: DataLoader 工作线程数，默认 16
        --loader_shuffle: 是否打乱数据顺序，默认 True
        --pin_memory: 是否使用 pin_memory，默认 False
        --prefetch_factor: 预取因子，默认 3
        --device: 训练设备，默认 'cuda'
        --batch_size: 批次大小，默认 4（分布式训练时为单卡批次）
        --n_epochs: 训练轮数，默认 100
        --lr: 学习率，默认 1e-5
        --lr_scheduler: 是否使用学习率调度器，默认 False
        --local-rank: 本地进程排名（分布式训练自动设置），默认 0
        --hidden_size: 隐藏层维度，默认 512
        --hardtanh_limit: HardTanh 激活函数的边界值，默认 100
        --use_image: 是否使用图像特征，默认 False
        --image_network: 图像主干网络类型，默认 'resnet50'
        --use_attribute: 是否使用属性特征，默认 True
        --use_opticalflow: 是否使用光流特征，默认 True
    """
    parser = argparse.ArgumentParser(description='Train PTINet network')
    
    # 数据路径配置
    parser.add_argument('--data_dir', type=str,
                        default='/home/farzeen/work/aa_postdoc/intent/JAAD/PN/',
                        required=False, help='数据集目录路径')
    parser.add_argument('--dataset', type=str, 
                        default='pie',
                        required=False, help='数据集名称（jaad/pie）')
    parser.add_argument('--out_dir', type=str, 
                        default='/home/farzeen/work/aa_postdoc/intent/PIE_bbox_image/bounding-box-prediction/output',
                        required=False, help='输出目录路径')  
    
    # 数据序列配置
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

    # 数据加载/保存配置
    parser.add_argument('--dtype', type=str, default='train', help='数据类型')
    parser.add_argument("--from_file", type=bool, default=False, help='是否从文件加载预处理数据')       
    parser.add_argument('--save', type=bool, default=True, help='是否保存训练结果')
    parser.add_argument('--log_name', type=str, default='', help='日志文件夹名称')
    parser.add_argument('--loader_workers', type=int, default=16, help='DataLoader 工作线程数')
    parser.add_argument('--loader_shuffle', type=bool, default=True, help='是否打乱数据顺序')
    parser.add_argument('--pin_memory', type=bool, default=False, help='是否使用 pin_memory')
    parser.add_argument('--prefetch_factor', type=int, default=3, help='预取因子')

    # 训练配置（分布式训练相关）
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--batch_size', type=int, default=4, help='单卡批次大小')
    parser.add_argument('--n_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')  # 修正类型为 float
    parser.add_argument('--lr_scheduler', type=bool, default=False, help='是否使用学习率调度器')
    parser.add_argument('--local-rank', type=int, default=0, help='本地进程排名（分布式训练自动设置）')

    # 网络配置
    parser.add_argument('--hidden_size', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--hardtanh_limit', type=int, default=100, help='HardTanh 激活函数边界')
    parser.add_argument('--use_image', type=bool, default=False,
                        help='使用图像作为输入特征')
    parser.add_argument('--image_network', type=str, default='resnet50',
                        help='选择图像主干网络（clstm/resnet50）')
    parser.add_argument('--use_attribute', type=bool, default=True,
                        help='使用属性作为输入特征')
    parser.add_argument('--use_opticalflow', type=bool, default=True,
                        help='使用光流作为输入特征')
    
    args = parser.parse_args()

    return args



def train(args, train_set, val_set):
    """
    PTINet 分布式训练函数
    
    该函数执行完整的分布式训练流程：
    1. 初始化分布式训练环境（NCCL backend）
    2. 初始化模型、优化器和损失函数
    3. 使用 DistributedSampler 进行分布式数据采样
    4. 迭代训练：前向传播 -> 计算损失 -> 反向传播 -> 参数更新
    5. 每轮训练后进行验证，计算评估指标
    6. 保存最佳模型和训练数据（仅在 global_rank==0 上执行）
    
    Args:
        args (argparse.Namespace): 命令行参数，包含训练配置
        train_set (Dataset): 训练数据集对象
        val_set (Dataset): 验证数据集对象
    
    分布式训练说明：
        - 使用 NCCL 作为通信后端
        - 使用 DistributedSampler 确保每个进程处理不同的数据子集
        - 使用 DistributedDataParallel 包装模型进行多卡并行
        - 仅在 global_rank==0 的进程上打印日志和保存模型
    
    关键指标说明:
        ADE (Average Displacement Error): 平均位移误差，衡量轨迹预测精度
        FDE (Final Displacement Error): 最终位移误差，衡量终点预测精度
        AIOU/FIOU: 平均/最终 IoU，衡量预测轨迹与真实轨迹的重合度
        intention_acc: 行人穿越意图分类准确率
    """
    print('='*100)
    print('Training ...')
    print('Learning rate: ' + str(args.lr))
    print('Number of epochs: ' + str(args.n_epochs))
    print('Hidden layer size: ' + str(args.hidden_size) + '\n')


    # 初始化分布式训练环境
    # backend='nccl'：使用 NCCL 作为 GPU 间通信后端（NVIDIA 推荐）
    dist.init_process_group(backend='nccl')
    
    # 设置随机种子保证可复现性（所有进程使用相同种子）
    torch.manual_seed(0)
    
    # 获取本地进程排名（从环境变量读取，由 torch.distributed.launch 设置）
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # 设置当前进程使用的 GPU 设备
    torch.cuda.set_device(local_rank)
    
    # 启用 cuDNN 加速
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # 仅在 global_rank==0 的进程上打印日志（避免重复输出）
    verbose = dist.get_rank() == 0
    
    # 初始化 PTINet 模型并移至当前 GPU
    net = network.PTINet(args).cuda()
    
    # 使用 DistributedDataParallel 包装模型
    # device_ids=[local_rank]：指定当前进程使用的 GPU
    # find_unused_parameters=True：允许存在未使用的参数（某些模块可能不参与计算）
    net = DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=True)


    # Enable Tensor Core operations (注释掉，如需使用混合精度可取消注释)
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)


    # 配置优化器：Adam + L2正则化（weight_decay=1e-7）
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-7)


    # 配置学习率调度器（可选）：每30轮学习率减半
    if args.lr_scheduler:
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, 
        #                                                 threshold = 1e-8, verbose=True)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
    
    # 创建分布式数据采样器（确保各进程数据不重复）
    train_sampler = DistributedSampler(train_set)     
    
    # 创建训练数据加载器（使用分布式采样器）
    dataloader_train = torch.utils.data.DataLoader(
        train_set, 
        batch_size=args.batch_size,
        pin_memory=args.pin_memory, 
        num_workers=0,  # 分布式训练中通常设置为0，避免多进程数据加载冲突
        drop_last=True,
        sampler=train_sampler
    )
    
    # 定义损失函数
    mse = nn.MSELoss()       # 速度预测损失：均方误差
    # huber = torch.nn.HuberLoss(reduction='sum', delta=1.0)  # Huber损失（更鲁棒）
    bce = nn.BCELoss()       # 穿越状态预测损失：二元交叉熵
    
    data = []                # 存储每轮训练数据（用于后续分析）
    best_ade = float('inf')  # 最佳 ADE 指标（用于保存最优模型）
    writer = SummaryWriter() # TensorBoard 日志记录器

    for epoch in range(args.n_epochs):
        """
        训练轮次循环（分布式版本）
        
        每轮训练包含两个阶段：
        1. 训练阶段：遍历训练集，更新模型参数
        2. 验证阶段：遍历验证集，计算评估指标
        
        损失函数设计：
            total_loss = speed_loss + crossing_loss + mloss
            - speed_loss: 速度预测损失（MSE）
            - crossing_loss: 穿越状态预测损失（BCE，多帧平均）
            - mloss: 模型内部损失（如VAE的KL散度等）
        """
        start = time.time()
        
        # 初始化训练/验证损失累计变量
        avg_epoch_train_s_loss = 0  # 训练速度损失
        avg_epoch_val_s_loss   = 0  # 验证速度损失
        avg_epoch_train_c_loss = 0  # 训练穿越损失
        avg_epoch_val_c_loss   = 0  # 验证穿越损失
        avg_epoch_train_t_loss = 0  # 训练总损失
        avg_epoch_val_v_loss   = 0  # 验证模型内部损失
        
        # 初始化验证指标累计变量
        ade  = 0  # 平均位移误差
        fde  = 0  # 最终位移误差
        aiou = 0  # 平均IoU
        fiou = 0  # 最终IoU
        avg_acc = 0   # 状态分类准确率
        avg_rec = 0   # 状态分类召回率
        avg_pre = 0   # 状态分类精确率
        mAP = 0       # 平均精度均值
        
        counter = 0
        for idx, inputs in enumerate(dataloader_train):
            counter += 1
            # 将数据移至当前GPU（non_blocking=True 提升数据传输效率）
            speed = inputs['speed'].cuda(non_blocking=True)              # 历史速度序列 [B, input_len, 2]
            future_speed = inputs['future_speed'].cuda(non_blocking=True) # 未来速度序列 [B, output_len, 2]
            pos = inputs['pos'].cuda(non_blocking=True)                  # 历史位置序列 [B, input_len, 2]
            future_pos = inputs['future_pos'].cuda(non_blocking=True)    # 未来位置序列 [B, output_len, 2]
            future_cross = inputs['future_cross'].cuda(non_blocking=True)# 未来穿越状态 [B, output_len, 2]
            optical = inputs['optical'].cuda(non_blocking=True)          # 光流特征
            ped_behavior = inputs['ped_behavior'].cuda(non_blocking=True)# 行人行为特征
            images = inputs['image'].cuda(non_blocking=True)             # 图像特征
            label_c = inputs['cross_label'].cuda(non_blocking=True)      # 穿越意图标签
            ped_attribute = inputs['ped_attribute'].cuda(non_blocking=True) # 行人属性特征
            scene_attribute = inputs['scene_attribute'].cuda(non_blocking=True) # 场景属性特征
            
            net.zero_grad()  # 清空梯度
            
            # 前向传播：average=False 表示训练模式，不计算意图概率
            mloss, speed_preds, crossing_preds = net(
                speed=speed, 
                pos=pos,
                ped_attribute=ped_attribute,
                ped_behavior=ped_behavior,
                scene_attribute=scene_attribute,
                images=images,
                optical=optical,
                average=False
            )

            # 计算速度预测损失（除以100进行归一化）
            speed_loss = mse(speed_preds, future_speed) / 100

            # 计算穿越状态预测损失（多帧BCE损失的平均）
            crossing_loss = 0
            for i in range(future_cross.shape[1]):
                crossing_loss += bce(crossing_preds[:,i], future_cross[:,i])
            crossing_loss /= future_cross.shape[1]  # 按帧数平均
            
            # 总损失 = 速度损失 + 穿越损失 + 模型内部损失
            loss = speed_loss + crossing_loss + mloss
            loss.backward()  # 反向传播
            optimizer.step() # 更新参数
            
            # 累加损失值
            avg_epoch_train_s_loss += float(speed_loss)
            avg_epoch_train_c_loss += float(crossing_loss)
            avg_epoch_train_t_loss += float(loss)
            torch.cuda.synchronize()  # 等待GPU操作完成

        # 计算平均训练损失
        avg_epoch_train_s_loss /= counter
        avg_epoch_train_c_loss /= counter
        avg_epoch_train_t_loss /= counter
        
        # 记录训练损失到TensorBoard
        writer.add_scalar("Loss_speed/train", avg_epoch_train_s_loss, epoch)
        writer.add_scalar("Loss_crossing/train", avg_epoch_train_c_loss, epoch)
        writer.add_scalar("Loss/train", avg_epoch_train_t_loss, epoch)

        # 创建验证集分布式采样器
        val_sampler = DistributedSampler(val_set)     
        
        # 创建验证数据加载器（使用分布式采样器）
        dataloader_val = torch.utils.data.DataLoader(
            val_set, 
            batch_size=args.batch_size,
            pin_memory=args.pin_memory, 
            num_workers=0, 
            drop_last=True,
            sampler=val_sampler
        )

        counter = 0
        # 初始化预测结果和标签存储列表
        state_preds = []    # 穿越状态预测结果
        state_targets = []  # 穿越状态真实标签
        intent_preds = []   # 意图预测结果
        intent_targets = [] # 意图真实标签
        f1_sc = []          # 每批次F1分数
        pre = []            # 每批次精确率
        recall_sc = []      # 每批次召回率
        acc = []            # 每批次准确率


        for idx, val_in in enumerate(dataloader_val):
            """
            验证阶段：遍历验证集计算指标（分布式版本）
            
            验证阶段与训练阶段的主要区别：
            1. average=True：计算意图概率（用于意图分类评估）
            2. torch.no_grad()：禁用梯度计算，节省内存并加速推理
            3. 计算更多评估指标（ADE, FDE, IoU, 分类指标等）
            """
            counter += 1

            # 将验证数据移至当前GPU
            speed = val_in['speed'].cuda(non_blocking=True)              # 历史速度序列
            future_speed = val_in['future_speed'].cuda(non_blocking=True) # 未来速度序列
            pos = val_in['pos'].cuda(non_blocking=True)                  # 历史位置序列
            future_pos = val_in['future_pos'].cuda(non_blocking=True)    # 未来位置序列
            future_cross = val_in['future_cross'].cuda(non_blocking=True)# 未来穿越状态
            ped_attribute = val_in['ped_attribute'].cuda(non_blocking=True)
            scene_attribute = val_in['scene_attribute'].cuda(non_blocking=True)
            optical = val_in['optical'].cuda(non_blocking=True)          # 光流特征
            ped_behavior = val_in['ped_behavior'].cuda(non_blocking=True)# 行人行为特征
            images = val_in['image'].cuda(non_blocking=True)             # 图像特征
            label_c = val_in['cross_label'].cuda(non_blocking=True)      # 穿越意图标签
            
            with torch.no_grad():
                # 前向传播：average=True 表示验证模式，计算意图概率
                vloss, speed_preds, crossing_preds, intentions = net(
                    speed=speed, 
                    pos=pos,
                    ped_attribute=ped_attribute,
                    ped_behavior=ped_behavior,
                    scene_attribute=scene_attribute,
                    images=images,
                    optical=optical,
                    average=True
                )
                
                # 计算验证损失
                speed_loss_v = mse(speed_preds, future_speed) / 100
                
                crossing_loss_v = 0
                for i in range(future_cross.shape[1]):
                    crossing_loss_v += bce(crossing_preds[:,i], future_cross[:,i])
                crossing_loss_v /= future_cross.shape[1]
                
                # 累加验证损失
                avg_epoch_val_s_loss += float(speed_loss_v)
                avg_epoch_val_c_loss += float(crossing_loss_v)
                avg_epoch_val_v_loss += float(vloss)
                
                # 将速度预测转换为位置预测（使用历史位置作为初始条件）
                preds_p = utils.speed2pos(speed_preds, pos)
                # 计算轨迹预测指标
                ade += float(utils.ADE(preds_p, future_pos))   # 平均位移误差
                fde += float(utils.FDE(preds_p, future_pos))   # 最终位移误差
                aiou += float(utils.AIOU(preds_p, future_pos)) # 平均IoU
                fiou += float(utils.FIOU(preds_p, future_pos)) # 最终IoU
                
                # 处理穿越状态预测结果
                # future_cross[:,:,1]：取穿越状态的正类概率（是否穿越）
                future_cross_np = future_cross[:,:,1].view(-1).cpu().numpy()
                # 对穿越状态预测取argmax得到二分类结果
                crossing_preds_np = np.argmax(crossing_preds.view(-1,2).detach().cpu().numpy(), axis=1)
                # 计算分类指标
                precision, recall, f1, accuracy = calculate_score(crossing_preds_np, future_cross_np)
                pre.append(precision)
                recall_sc.append(recall)
                f1_sc.append(f1)
                acc.append(accuracy)
                
                # 处理意图预测结果
                label_c_np = label_c.view(-1).cpu().numpy()
                intentions_np = intentions.view(-1).detach().cpu().numpy()

                # 收集所有预测结果用于后续计算总体指标
                state_preds.extend(crossing_preds_np)
                state_targets.extend(future_cross_np)
                intent_preds.extend(intentions_np)
                intent_targets.extend(label_c_np)
                torch.cuda.synchronize()

            
        # 计算平均验证损失
        avg_epoch_val_s_loss /= counter
        avg_epoch_val_c_loss /= counter
        
        # 计算平均验证指标
        ade  /= counter
        fde  /= counter     
        aiou /= counter
        fiou /= counter

        # 验证总损失
        v_loss = avg_epoch_val_s_loss + avg_epoch_val_c_loss + avg_epoch_val_v_loss

        # 记录验证损失到TensorBoard
        writer.add_scalar("Loss_speed/val", avg_epoch_val_s_loss, epoch)
        writer.add_scalar("Loss_crossing/val", avg_epoch_val_c_loss, epoch)


        # 计算意图分类指标（所有样本的平均值）
        pre_int, recall_int, f1_intt, acc_int = calculate_score(np.array(intent_preds), np.array(intent_targets))
        # 计算穿越状态分类指标（按批次平均）
        pre = np.sum(pre) / counter
        recall_sc = np.sum(recall_sc) / counter
        f1_sc = np.sum(f1_sc) / counter
        acc = np.sum(acc) / counter

        # 使用sklearn计算总体指标
        avg_acc = accuracy_score(state_targets, state_preds)          # 状态分类准确率
        f1_state = f1_score(state_targets, state_preds)               # 状态分类F1
        avg_rec = recall_score(state_targets, state_preds, average='binary', zero_division=1)  # 状态分类召回率
        avg_pre = precision_score(state_targets, state_preds, average='binary', zero_division=1)  # 状态分类精确率
        mAP = average_precision_score(state_targets, state_preds, average=None)  # 状态分类mAP
        intent_acc = accuracy_score(intent_targets, intent_preds)    # 意图分类准确率
        f1_int = f1_score(intent_targets, intent_preds)              # 意图分类F1
        intent_mAP = average_precision_score(intent_targets, intent_preds, average=None)  # 意图分类mAP
        
        # 保存本轮训练数据（用于后续分析和可视化）
        data.append([epoch, avg_epoch_train_s_loss, avg_epoch_val_s_loss, \
                    avg_epoch_train_c_loss, avg_epoch_val_c_loss, \
                    ade, fde, aiou, fiou, intent_acc])

        # 更新学习率（如果启用了调度器）
        if args.lr_scheduler:
            scheduler.step(avg_epoch_train_t_loss)

        # 保存最佳模型（基于ADE指标）
        if ade < best_ade:
            """
            保存最佳模型（分布式版本）
            
            当当前轮次的ADE小于历史最佳ADE时，保存当前模型参数。
            模型文件名格式：model_best_{lr}_{hidden_size}[_scheduler].pkl
            - lr: 学习率
            - hidden_size: 隐藏层维度
            - _scheduler: 可选后缀，表示使用了学习率调度器
            
            注意：在分布式训练中，由于所有进程的模型参数是同步的，
            理论上每个进程都可以保存模型。但通常只在 global_rank==0 上保存。
            """
            best_ade = ade
            file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
            if args.lr_scheduler:
                 modelname = 'model_best' + file + '_scheduler.pkl'
            else:
                 modelname = 'model_best' + file + '.pkl'   
            torch.save(net.state_dict(), os.path.join(args.out_dir, args.log_name, modelname))
        
        # 打印本轮训练结果
        print('e:', epoch, 
             '| ade: %.4f'% ade, 
            '| fde: %.4f'% fde, '| aiou: %.4f'% aiou, '| fiou: %.4f'% fiou, '| state_acc: %.4f'% avg_acc, acc, 
            '| intention_acc: %.4f'% intent_acc, acc_int, '| f1_int: %.4f'% f1_int, f1_intt, 
            '| f1_state: %.4f'% f1_state, f1_sc, '| pre: %.4f'% pre, '| recall_sc: %.4f'% recall_sc,
            '| pre_int: %.4f'% pre_int, '| recall_int: %.4f'% recall_int)
   

    # 将训练数据转换为DataFrame（便于后续分析和可视化）
    df = pd.DataFrame(data, columns=['epoch', 'train_loss_s', 'val_loss_s', 'train_loss_c', 'val_loss_c',
                                     'ade', 'fde', 'aiou', 'fiou', 'intention_acc']) 

    # 保存最终训练结果
    if args.save:
        print('\nSaving ...')
        file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
        if args.lr_scheduler:
            filename = 'data_final' + file + '_scheduler.csv'
            modelname = 'model_final' + file + '_scheduler.pkl'
        else:
            filename = 'data_final' + file + '.csv'
            modelname = 'model_final' + file + '.pkl'

        # 保存训练数据CSV文件
        df.to_csv(os.path.join(args.out_dir, args.log_name, filename), index=False)
        # 保存最终模型
        torch.save(net.state_dict(), os.path.join(args.out_dir, args.log_name, modelname))
        
        print('Training data and model saved to {}\n'.format(os.path.join(args.out_dir, args.log_name)))

    print('='*100)
    print('Done !')


if __name__ == '__main__':
    """
    分布式训练脚本入口函数
    
    执行流程：
    1. 解析命令行参数和配置文件
    2. 加载训练集和验证集
    3. 启动分布式训练
    
    分布式训练启动方式：
        使用 torch.distributed.launch 或 torchrun 启动：
        python -m torch.distributed.launch --nproc_per_node=<num_gpus> train.py
        或
        torchrun --nproc_per_node=<num_gpus> train.py
    
    注意：分布式训练中，每个进程都会执行此入口函数，
    但分布式环境初始化在 train() 函数内部进行。
    """
    # 打印当前时间（用于记录训练开始时间）
    print("Date and time:", datetime.datetime.now())

    # 解析命令行参数
    args = parse_args()
    # 解析配置文件（如果配置文件中禁用了命令行参数，则使用配置文件的值）
    config = parse_config_file('/home/lbh/PTINet/config.yml')
    if config.get('use_argument_parser') == False:
        # Override command-line arguments with values from the configuration file
        for arg in vars(args):
            if arg in config:
                setattr(args, arg, config[arg])
    print(args)

    # 加载训练数据集
    train_set = eval('datasets.' + args.dataset)(
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

    # 加载验证数据集
    val_set = eval('datasets.' + args.dataset)(
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

    # 启动分布式训练
    train(args, train_set, val_set)
    