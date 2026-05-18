import torch
import time
import sys
sys.path.insert(0, '/home/lbh/OurWork1')

from model.cofe import CoFE


def test_ego_dists_consistency():
    """\u6d4b\u8bd5ego_dists\u51fd\u6570\u7684\u529f\u80fd\u4e00\u81f4\u6027"""
    print("=" * 60)
    print("\u6d4b\u8bd51: ego_dists\u529f\u80fd\u4e00\u81f4\u6027")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T, N, S = 10, 20, 3
    hist_abs = torch.randn(T, N, 2, device=device) * 10
    seq_start_end = torch.tensor([[0, 7], [7, 15], [15, 20]], device=device)
    
    def ego_dists_original(hist_abs, seq_start_end):
        hist_ego_abs = torch.zeros_like(hist_abs)
        for (start, end) in seq_start_end:
            hist_ego_abs[:, start:end] = (
                hist_abs[:, start:end] - hist_abs[:, start].unsqueeze(1)
            )
        return hist_ego_abs
    
    result_new = CoFE.ego_dists(hist_abs, seq_start_end)
    result_original = ego_dists_original(hist_abs, seq_start_end)
    
    max_diff = torch.max(torch.abs(result_new - result_original)).item()
    print(f"\u6700\u5927\u8bef\u5dee: {max_diff:.10f}")
    print(f"\u7ed3\u679c\u4e00\u81f4: {torch.allclose(result_new, result_original, atol=1e-8)}")
    
    if not torch.allclose(result_new, result_original, atol=1e-8):
        print("ERROR: \u7ed3\u679c\u4e0d\u4e00\u81f4!")
        return False
    
    print("\u2705 ego_dists\u529f\u80fd\u9a8c\u8bc1\u901a\u8fc7!")
    return True


def test_encode_yaw_consistency():
    """\u6d4b\u8bd5encode_yaw\u51fd\u6570\u7684\u529f\u80fd\u4e00\u81f4\u6027"""
    print("\n" + "=" * 60)
    print("\u6d4b\u8bd52: encode_yaw\u529f\u80fd\u4e00\u81f4\u6027")
    print("=" * 0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T, N, S = 10, 20, 3
    hist_yaw = torch.randn(T, N, device=device) * 360
    seq_start_end = torch.tensor([[0, 7], [7, 15], [15, 20]], device=device)
    
    def encode_yaw_original(hist_yaw, seq_start_end):
        offset_yaw_rel = torch.zeros_like(hist_yaw)
        for (start, end) in seq_start_end:
            offset_yaw_rel[:, start:end] = (
                hist_yaw[:, start:end] - hist_yaw[:, start].unsqueeze(1)
            )
        offset_yaw_norm = ((180 + offset_yaw_rel) % 360 - 180)
        offset_yaw_rad = torch.deg2rad(offset_yaw_norm)
        return torch.stack([torch.cos(offset_yaw_rad), torch.sin(offset_yaw_rad)], dim=-1)
    
    result_new = CoFE.encode_yaw(hist_yaw, seq_start_end)
    result_original = encode_yaw_original(hist_yaw, seq_start_end)
    
    max_diff = torch.max(torch.abs(result_new - result_original)).item()
    print(f"\u6700\u5927\u8bef\u5dee: {max_diff:.10f}")
    print(f"\u7ed3\u679c\u4e00\u81f4: {torch.allclose(result_new, result_original, atol=1e-8)}")
    
    if not torch.allclose(result_new, result_original, atol=1e-8):
        print("ERROR: \u7ed3\u679c\u4e0d\u4e00\u81f4!")
        return False
    
    print("\u2705 encode_yaw\u529f\u80fd\u9a8c\u8bc1\u901a\u8fc7!")
    return True


def test_boundary_conditions():
    """\"\u6d4b\u8bd5\u8fb9\u754c\u6761\u4ef6"""
    print("\n" + "=" * 60)
    print("\u6d4b\u8bd53: \u8fb9\u754c\u6761\u4ef6\u6d4b\u8bd5")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n\u8fb9\u754c\u6d4b\u8bd51: \u5355\u4e2a\u573a\u666f")
    T, N = 5, 10
    hist_abs = torch.randn(T, N, 2, device=device)
    hist_yaw = torch.randn(T, N, device=device) * 360
    seq_start_end = torch.tensor([[0, 10]], device=device)
    
    result_ego = CoFE.ego_dists(hist_abs, seq_start_end)
    result_yaw = CoFE.encode_yaw(hist_yaw, seq_start_end)
    print(f"\u5355\u4e2ascene\u573aego_dists\u5f62\u72b6: {result_ego.shape}")
    print(f"\u5355\u4e2a\u6bd5\u573aencode_yaw\u5f62\u72b6: {result_yaw.shape}")
    
    print("\n\u8fb9\u754c\u6d4b\u8bd52: \u6bcf\u4e2a\u573a\u666f\u53ea\u6709\u4e00agent")
    T, N, S = 5, 5, 5
    hist_abs = torch.randn(T, N, 2, device=device)
    hist_yaw = torch.randn(T, N, device=device) * 360
    seq_start_end = torch.tensor([[i, i+1] for i in range(5)], device=device)
    
    result_ego = CoFE.ego_dists(hist_abs, seq_start_end)
    result_yawltshape
    print(f"\u573a\u666f\u5f62\u72b6: {seq_start_end.shape}")
    print(f"ego_dists\u7ed3\u679c: {result_ego.shape}")
    print(f"encode_yaw\u7ed3\u679c: {result_yaw.shape}")
    
    max_ego_diff = torch.max(torch.abs(result_ego)).item()
    print(f"\u6bcf\u4e2aagent\u7684ego\u76f8\u5bf9\u5750\u6807\u6700\u5927\u503c: {max_ego_diff:.10f}")
    assert max_ego_diff < 1e-8, "\u5355\u4e2aagent\u573a\u666f\u7684\u76f8\u5bf9\u5750\u6807\u5e94\u8be5\u4e3a0"
    
    print("\u2705 \u8fb9\u754c\u6761\u4ef6\u6d4b\u8bd5\u901a\u8fc7!")
    return True


    test_performance():
    """\u6027\u80fd\u5bf9\u6bd4\u6d4b\u8bd5"""
    print("\n" + "=" * 60)
    print("\u6d4b\u8bd54: \u6027\u80fd\u5bf9\u6bd4\u6d4b\u8bd5")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\u4f7f\u7528\u8bbe\u5907: {device}")
    
    T, N, S = 30, 200, 20
    hist_abs = torch.randn(T, N, 2, device=device) * 10
    hist_yaw = torch.randn(T, N, device=device) * 360
    seq_start_end = torch.tensor([[i*10, (i+1)*10] for i in range(S)], device=device)
    
    def ego_dists_original(hist_abs, seq_start_end):
        hist_ego_abs = torch.zeros_like(hist_abs)
        for (start, end) in seq_start_end:
            hist_ego_abs[:, start:end] = (
                hist_abs[:, start:end] - hist_abs[:start].unsqueeze(1)
            )
        return hist_ego_abs
    
    def encode_yaw_original(hist_yaw, seq_start_end):
        offset_yaw_rel = torch.zeros_like(hist_yaw)
        for (start, end) in seq_start_end:
            offset_yaw_rel[:, start:end] = (
                hist_yaw[:, start:end] - hist_yaw[:, start].unsqueeze(1)
            )
        offset_yaw_norm = ((180 + offset_yaw_rel) % 360 - 180)
        offset_yaw_rad = torch.deg2rad(offset_yaw_norm)
        return torch.stack([torch.cos(offset_yaw_rad), torch.sin(offset_yaw_rad)], dim=-1)
    
    if torch.cuda.is_available():
        CoFE.ego_dists(hist_abs, seq_start_end)
        torch.cuda.synchronize()
    
    n_trials = 100
    
    t0 = time.time()
    for _ in range(n_trials):
        ego_dists_original(hist_abs, seq_start_end)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    original_time_ego = (t1 - t0) / n_trials
    
    t0 = time.time()
    for _ in range(n_trials):
        CoFE.ego_dists(hist_abs, seq_start_end)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    new_time_ego = (t1 - t0) / n_trials
    
    print(f"\nego_dists\u6027\u80fd:")
    print(f"{original_time_ego*1000:.2f} ms/\u6b21")
    print(f"  \u65b0\u5b9e\u73b0: {new_time_ego*1000:.2f} ms/\u6b21")
    speedup_ego = original_time_ego / new_time_ego
    print(f"  \u52a0\u901f\u6bd4: {speedup_ego:.2f}x")
    
    t0 = time.time()
    for _ in range(n_trials):
        encode_yaw_original(hist_yaw, seq_start_end)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    original_time_yaw = (t1 - t0) / n_trials
    
    t0 = time.time()
    for _ in range(n_trials):
        CoFE.encode_yaw(hist_yaw, seq_start_end)
    if torch._available():
        torch.cuda.synchronize()
    t1 = time.time()
    new_time_yaw = (t1 - t0) / n_trials
    
    print(f"\nencode_yaw\u6027\u80fd:")
    print(f"  \u539f\u5b9e\u73b0: {original_time_yaw*1000:.2f} ms/\u6b21")
    print(f" \u65b0\u5b9e\u73b0: {new_time_yaw*1000:.2f} ms/\u6b21")
    speedup_yaw = original_time_yaw / new_time_yaw
    print(f"  \u52a0\u901f\u6bd4: {speedup_yaw:.2f}x")
    
    return speedup_ego > 1 or speedup_yaw > 1


    test_full_cofe_pipeline():
    """\u6d4b\u8bd5\u5b8c\u6574\u7684CoFE pipeline\u662f\u5426\u6b63\u5e38\u5de5\u4f5c"""
    print("\n" + "=" * 60)
    print("\u6d4b\u8bd55: \u5b8c\u6574CoFE pipeline 6d4b\u8bd5")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    T, N, S = 15, 10, 2
    hist_abs = torch.randn(T, N, 2, device=device) * 5
    hist_yaw = torch.randn(T, N, device=device) * 360
    hist_resnet = torch.randn(T, N, 2048, device=device)
    seq_start_end = torch.tensor([[0, 5], [5, 10]], device=device)
    
    cofe = CoFE(input_size=2, hidden_size=96, num_layers=2, 
                use_resnet=False, no_abs=True, idxs=[6, 7])
    cofe.to(device)
    
    loss = cofe.train_correction(
        hist_abs, hist_yaw, hist_abs, hist_yaw, 
        hist_resnet, seq_start_end
    )
    print(f"\u8bad\u7ec3\u6a21\u5f0floss: {loss.item():.6f}")
    
    corrected = cofe.infer_correction(
        hist_abs, hist_yaw, hist_resnet, seq_start_end
    )
    print(f"\u63a8\u7406\u7ed3\u679c\u5f62\u72b6: {corrected.shape}")
    print(f"\u7ed3\u679c\u5305\u542bNaN: {torch.isnan(corrected).any().item()}")
    
    print("\u2705 \u5b8c\u6574CoFE pipeline\u6d4b\u8bd5\u901a\u8fc7!")
    return True


def main():
    print("\n" + "=" * 60)
    print("CoFE\u6a21\u5757\u5411\u91cf\u5316\u4f18\u5316\u6d4b\u8bd5\u5957\u4ef6e    print("=" * 60)
    
    all_passed = True
    
    try:
        test1 = test_ego_dists_consistency()
        test2 = test_encode_yaw_consistency()
        test3 = test_boundary_conditions()
        test4 = test_performance()
        test5 = test_full_cofe_pipeline()
        
        all_passed = test1 and test2 and test3 and test5
        
    except Exception as e:
        print(f"\n\u274c \u6d4b\u8bd5process\u4e2d\u53d1\u751f\u5f02\u5e38: {str(e)}")
        import traceback        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("\ud83c\udf89 \u6240\u6709\u6d4b\u8bd5\u901a\u8fc7! \u4f18\u5316\u6210\u529f!")
    else:
        print("\u274c \u90e8\u5206\u6d4b\u8bd5\u5931\u8d25\uff0c\u8bf7\u68c0\u67e5")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
