import torch


# TODO: 重构为DDP兼容版本后恢复使用（当前train.py直接使用DistributedSampler创建DataLoader）
def data_loader(args, dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    return dataloader


def ADE(pred, true, is_3D=False):
    if not is_3D:
        displacement = torch.sqrt((pred[:,:,0]-true[:,:,0])**2 + (pred[:,:,1]-true[:,:,1])**2)
    else:
        displacement = torch.sqrt((pred[:,:,0]-true[:,:,0])**2 + (pred[:,:,1]-true[:,:,1])**2\
                                  + (pred[:,:,2]-true[:,:,2])**2)
        
    ade = torch.mean(displacement)
    
    return ade


def FDE(pred, true, is_3D=False):
    if not is_3D:
        displacement = torch.sqrt((pred[:,-1,0]-true[:,-1,0])**2 + (pred[:,-1,1]-true[:,-1,1])**2)
    else:
        displacement = torch.sqrt((pred[:,-1,0]-true[:,-1,0])**2 + (pred[:,-1,1]-true[:,-1,1])**2\
                                  + (pred[:,-1,2]-true[:,-1,2])**2)
    
    fde = torch.mean(displacement)
    
    return fde


def speed2pos(preds, pos):
    dim = preds.shape[-1]
    pred_pos = torch.zeros(preds.shape[0], preds.shape[1], dim, device=preds.device)
    pred_pos[:, 0, :] = pos[:, -1, :] + preds[:, 0, :]
    for i in range(1, pred_pos.shape[1]):
        pred_pos[:, i, :] = pred_pos[:, i - 1, :] + preds[:, i, :]
    return pred_pos


def check_continuity(my_list, skip):
    return any(a+skip != b for a, b in zip(my_list, my_list[1:]))