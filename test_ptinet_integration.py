import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.network_image import PTINet


def test_ptinet_integration():
    """测试 PTINet + CoFE 的 T2FPV 前向集成。"""
    print("=" * 60)
    print("PTINet + CoFE 集成测试")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    class Args:
        def __init__(self):
            self.dataset = "T2FPV"
            self.hidden_size = 512
            self.device = device
            self.use_attribute = False
            self.use_image = True
            self.image_network = "clstm"
            self.use_opticalflow = False
            self.hardtanh_limit = 100
            self.use_cofe = True
            self.cofe_hidden_size = 96
            self.cofe_num_layers = 2
            self.cofe_use_resnet = False
            self.output = 10
            self.skip = 1

    args = Args()

    print("\n初始化 PTINet + CoFE...")
    model = PTINet(args).to(device)
    model.eval()
    print("✅ PTINet + CoFE 初始化成功")

    print("\n测试 FPV 模式前向传播...")
    T, N = 10, 8
    hist_all = torch.randn(T, N, 7, device=device) * 5
    hist_resnet = torch.randn(T, N, 2048, device=device)
    seq_start_end = torch.tensor([[0, 4], [4, 8]], device=device)

    with torch.no_grad():
        result = model(
            hist_all=hist_all,
            hist_resnet=hist_resnet,
            hist_seq_start_end=seq_start_end,
        )

    print(f"输出数量: {len(result)}")
    assert len(result) == 2
    print(f"轨迹预测形状: {result[1].shape}")
    assert result[1].shape == (N, args.output // args.skip, 2)
    assert not torch.isnan(result[1]).any()

    print("✅ FPV 模式前向传播成功!")
    print("\n" + "=" * 60)
    print("🎉 PTINet + CoFE 集成测试通过!")
    print("=" * 60)



def main():
    try:
        test_ptinet_integration()
        return 0
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


def test_t2fpv_batch_first_and_cofe_gradients():
    """验证 T2FPV batch-first 输入可用，并且端到端损失会更新 CoFE 参数。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Args:
        def __init__(self):
            self.dataset = "T2FPV"
            self.input = 5
            self.hidden_size = 32
            self.device = device
            self.use_attribute = False
            self.use_image = False
            self.image_network = "clstm"
            self.use_opticalflow = False
            self.hardtanh_limit = 100
            self.use_cofe = True
            self.cofe_hidden_size = 16
            self.cofe_num_layers = 1
            self.cofe_use_resnet = False
            self.cofe_loss_weight = 0.1
            self.output = 3
            self.skip = 1

    model = PTINet(Args()).to(device)
    model.train()

    batch, timesteps = 4, 5
    # 模拟 DataLoader 输出的 T2FPV batch-first 格式：(B, T, [x, y, yaw])。
    hist_all = torch.randn(batch, timesteps, 3, device=device)
    # 模拟可用于 CoFE 监督的干净历史轨迹，用于验证联合训练时 CoFE 有梯度。
    hist_abs_gt = hist_all[..., :2] + 0.01 * torch.randn(batch, timesteps, 2, device=device)
    target_speed = torch.randn(batch, 3, 2, device=device)

    mloss, speed_preds = model(hist_all=hist_all, hist_abs_gt=hist_abs_gt)
    loss = mloss + torch.nn.functional.mse_loss(speed_preds, target_speed)
    loss.backward()

    grad_norm = sum(
        p.grad.detach().abs().sum().item()
        for p in model.cofe.parameters()
        if p.grad is not None
    )
    assert grad_norm > 0


def test_t2fpv_resnet_cofe_without_features_uses_zero_fallback():
    """启用 CoFE ResNet 分支但缺少特征时，应使用零特征兜底而不是维度崩溃。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Args:
        def __init__(self):
            self.dataset = "T2FPV"
            self.input = 5
            self.hidden_size = 32
            self.device = device
            self.use_attribute = False
            self.use_image = False
            self.image_network = "clstm"
            self.use_opticalflow = False
            self.hardtanh_limit = 100
            self.use_cofe = True
            self.cofe_hidden_size = 16
            self.cofe_num_layers = 1
            self.cofe_use_resnet = True
            self.output = 3
            self.skip = 1

    model = PTINet(Args()).to(device)
    # 故意不传 hist_resnet：测试 CoFE ResNet 分支的零特征兜底逻辑。
    hist_all = torch.randn(2, 5, 3, device=device)

    with torch.no_grad():
        outputs = model(hist_all=hist_all)
    assert outputs[1].shape == (2, 3, 2)


if __name__ == "__main__":
    sys.exit(main())
