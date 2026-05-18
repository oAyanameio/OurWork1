import torch
import sys
sys.path.insert(0, '/home/lbh/OurWork1')

from model.network_image import PTINet


def test_ptinet_integration():
    """\u6d4b\u8bd5PTINet + CoFE\u7684\u96c6\u6210"""
    print("=" * 60)
    print("PTINet + CoFE\u96c6\u6210\u6d4b\u8bd5")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\u8bbe\u5907: {device}")
    
    class Args:
        def __init__(self):
            self.dataset = 'T2FPV'
            self.hidden_size = 512
            self.device = device
            self.use_attribute = False
            self.use_image = True
            self.image_network = 'clstm'
            self.use_opticalflow = False
            self.hardtanh_limit = 100
            self.use_cofe = True
            self.cofe_hidden_size = 96
            self.cofe_num_layers = 2
            self.cofe_use_resnet = False
            self.output = 10
            self.skip = 1
    
    args = Args()
    
    print("\n\u521d\u59cb\u5316PTINet + CoFE...")
    model = PTINet(args)
    model.to(device)
    print("\u2705 PTINet + CoFE\u521d\u59cb\u5316\u6210\u529f")
    
    print("\n\u6d4b\u8bd5FPV\u6a21\u5f0f\u524d\u5411\u4f20\u64ad...")
    T, N = 10, 8
    hist_all = torch.randn(T, N, 7, device=device) * 5
    hist_resnet = torch.randn(T, N, 2048, device=device)
    seq_start_end = torch.tensor([[0, 4], [4, 8]], device=device)
    
    with torch.no_grad():
        result = model(
            hist_all=hist_all, 
            hist_resnet=hist_resnet, 
            hist_seq_start_end=seq_start_end
        )
    
    print(f"\u8f93\u51fa\u6570\u91cf: {len(result)}")
    if len(result) >= 2:
        print(f"\u8f68\u8ff9\u9884\u6d4b\u5f62\u72b6: {result[1].shape}")
        print(f"\u610f\    result[2].shape}")
    
    print("\u2705 FPV\u6a21\u5f0f\u524d\u5411\u4f20\u64ad\u6210\u529f!")
    
    print("\n" + "=" * 60)
    print("\ud83c\udf89 PTINet + CoFE\u96c6\u6210\u6d4b\u8bd5\u901a\u8fc7!")
    print("=" * 60)
    
    return True    

def main():
    try:
        success = test_ptinet_integration()
        return 0 if success else 1
    except Exception as e:
        print(f"\n\u274c \u6d4b\u8bd5\u5931\u8d25: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
