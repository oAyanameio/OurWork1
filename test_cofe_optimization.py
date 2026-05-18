import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.cofe import CoFE


def test_ego_dists_consistency():
    """测试 ego_dists 向量化实现与原始循环实现一致。"""
    print("=" * 60)
    print("测试1: ego_dists 功能一致性")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, N = 10, 20
    hist_abs = torch.randn(T, N, 2, device=device) * 10
    seq_start_end = torch.tensor([[0, 7], [7, 15], [15, 20]], device=device)

    def ego_dists_original(hist_abs, seq_start_end):
        hist_ego_abs = torch.zeros_like(hist_abs)
        for start, end in seq_start_end:
            hist_ego_abs[:, start:end] = hist_abs[:, start:end] - hist_abs[:, start].unsqueeze(1)
        return hist_ego_abs

    result_new = CoFE.ego_dists(hist_abs, seq_start_end)
    result_original = ego_dists_original(hist_abs, seq_start_end)

    max_diff = torch.max(torch.abs(result_new - result_original)).item()
    print(f"最大误差: {max_diff:.10f}")
    assert torch.allclose(result_new, result_original, atol=1e-8)

    print("✅ ego_dists 功能验证通过!")


def test_encode_yaw_consistency():
    """测试 encode_yaw 向量化实现与原始循环实现一致。"""
    print("\n" + "=" * 60)
    print("测试2: encode_yaw 功能一致性")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, N = 10, 20
    hist_yaw = torch.randn(T, N, device=device) * 360
    seq_start_end = torch.tensor([[0, 7], [7, 15], [15, 20]], device=device)

    def encode_yaw_original(hist_yaw, seq_start_end):
        offset_yaw_rel = torch.zeros_like(hist_yaw)
        for start, end in seq_start_end:
            offset_yaw_rel[:, start:end] = hist_yaw[:, start:end] - hist_yaw[:, start].unsqueeze(1)
        offset_yaw_norm = ((180 + offset_yaw_rel) % 360 - 180)
        offset_yaw_rad = torch.deg2rad(offset_yaw_norm)
        return torch.stack([torch.cos(offset_yaw_rad), torch.sin(offset_yaw_rad)], dim=-1)

    result_new = CoFE.encode_yaw(hist_yaw, seq_start_end)
    result_original = encode_yaw_original(hist_yaw, seq_start_end)

    max_diff = torch.max(torch.abs(result_new - result_original)).item()
    print(f"最大误差: {max_diff:.10f}")
    assert torch.allclose(result_new, result_original, atol=1e-8)

    print("✅ encode_yaw 功能验证通过!")


def test_boundary_conditions():
    """测试单场景与单 agent 场景等边界条件。"""
    print("\n" + "=" * 60)
    print("测试3: 边界条件测试")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n边界测试1: 单个场景")
    T, N = 5, 10
    hist_abs = torch.randn(T, N, 2, device=device)
    hist_yaw = torch.randn(T, N, device=device) * 360
    seq_start_end = torch.tensor([[0, 10]], device=device)

    result_ego = CoFE.ego_dists(hist_abs, seq_start_end)
    result_yaw = CoFE.encode_yaw(hist_yaw, seq_start_end)
    assert result_ego.shape == (T, N, 2)
    assert result_yaw.shape == (T, N, 2)

    print("\n边界测试2: 每个场景只有一个 agent")
    T, N = 5, 5
    hist_abs = torch.randn(T, N, 2, device=device)
    hist_yaw = torch.randn(T, N, device=device) * 360
    seq_start_end = torch.tensor([[i, i + 1] for i in range(N)], device=device)

    result_ego = CoFE.ego_dists(hist_abs, seq_start_end)
    result_yaw = CoFE.encode_yaw(hist_yaw, seq_start_end)
    assert result_ego.shape == (T, N, 2)
    assert result_yaw.shape == (T, N, 2)

    max_ego_diff = torch.max(torch.abs(result_ego)).item()
    max_yaw_sin = torch.max(torch.abs(result_yaw[..., 1])).item()
    assert max_ego_diff < 1e-8, "单 agent 场景的 ego 相对坐标应该为 0"
    assert max_yaw_sin < 1e-6, "单 agent 场景的 ego 相对 yaw 应该为 0 度"

    print("✅ 边界条件测试通过!")


def test_performance():
    """性能对比测试；只用于输出参考，不作为正确性门禁。"""
    print("\n" + "=" * 60)
    print("测试4: 性能对比测试")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    T, N, S = 30, 200, 20
    hist_abs = torch.randn(T, N, 2, device=device) * 10
    hist_yaw = torch.randn(T, N, device=device) * 360
    seq_start_end = torch.tensor([[i * 10, (i + 1) * 10] for i in range(S)], device=device)

    def ego_dists_original(hist_abs, seq_start_end):
        hist_ego_abs = torch.zeros_like(hist_abs)
        for start, end in seq_start_end:
            hist_ego_abs[:, start:end] = hist_abs[:, start:end] - hist_abs[:, start].unsqueeze(1)
        return hist_ego_abs

    def encode_yaw_original(hist_yaw, seq_start_end):
        offset_yaw_rel = torch.zeros_like(hist_yaw)
        for start, end in seq_start_end:
            offset_yaw_rel[:, start:end] = hist_yaw[:, start:end] - hist_yaw[:, start].unsqueeze(1)
        offset_yaw_norm = ((180 + offset_yaw_rel) % 360 - 180)
        offset_yaw_rad = torch.deg2rad(offset_yaw_norm)
        return torch.stack([torch.cos(offset_yaw_rad), torch.sin(offset_yaw_rad)], dim=-1)

    if torch.cuda.is_available():
        CoFE.ego_dists(hist_abs, seq_start_end)
        torch.cuda.synchronize()

    n_trials = 20

    t0 = time.time()
    for _ in range(n_trials):
        ego_dists_original(hist_abs, seq_start_end)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    original_time_ego = (time.time() - t0) / n_trials

    t0 = time.time()
    for _ in range(n_trials):
        CoFE.ego_dists(hist_abs, seq_start_end)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    new_time_ego = (time.time() - t0) / n_trials

    print("\nego_dists 性能:")
    print(f"  原实现: {original_time_ego * 1000:.2f} ms/次")
    print(f"  新实现: {new_time_ego * 1000:.2f} ms/次")
    print(f"  加速比: {original_time_ego / new_time_ego:.2f}x")

    t0 = time.time()
    for _ in range(n_trials):
        encode_yaw_original(hist_yaw, seq_start_end)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    original_time_yaw = (time.time() - t0) / n_trials

    t0 = time.time()
    for _ in range(n_trials):
        CoFE.encode_yaw(hist_yaw, seq_start_end)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    new_time_yaw = (time.time() - t0) / n_trials

    print("\nencode_yaw 性能:")
    print(f"  原实现: {original_time_yaw * 1000:.2f} ms/次")
    print(f"  新实现: {new_time_yaw * 1000:.2f} ms/次")
    print(f"  加速比: {original_time_yaw / new_time_yaw:.2f}x")



def test_full_cofe_pipeline():
    """测试完整的 CoFE pipeline 是否正常工作。"""
    print("\n" + "=" * 60)
    print("测试5: 完整 CoFE pipeline 测试")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T, N = 15, 10
    hist_abs = torch.randn(T, N, 2, device=device) * 5
    hist_yaw = torch.randn(T, N, device=device) * 360
    hist_resnet = torch.randn(T, N, 2048, device=device)
    seq_start_end = torch.tensor([[0, 5], [5, 10]], device=device)

    cofe = CoFE(input_size=2, hidden_size=96, num_layers=2, use_resnet=False, no_abs=True, idxs=[6, 7]).to(device)

    loss = cofe.train_correction(hist_abs, hist_yaw, hist_abs, hist_yaw, hist_resnet, seq_start_end)
    print(f"训练模式 loss: {loss.item():.6f}")

    corrected = cofe.infer_correction(hist_abs, hist_yaw, hist_resnet, seq_start_end)
    print(f"推理结果形状: {corrected.shape}")
    assert corrected.shape == hist_abs.shape
    assert not torch.isnan(corrected).any()

    print("✅ 完整 CoFE pipeline 测试通过!")


def main():
    print("\n" + "=" * 60)
    print("CoFE 模块向量化优化测试套件")
    print("=" * 60)

    all_passed = True

    try:
        test_ego_dists_consistency()
        test_encode_yaw_consistency()
        test_boundary_conditions()
        test_performance()
        test_full_cofe_pipeline()

    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {str(e)}")
        import traceback

        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过! 优化成功!")
    else:
        print("❌ 部分测试失败，请检查")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
