import torch, sys
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("torch arch list:", torch.cuda.get_arch_list())
    try:
        x = torch.randn(2, 1, 8, 32, 32, device="cuda")
        c = torch.nn.Conv3d(1, 4, 3, padding=1).cuda()
        y = c(x); torch.cuda.synchronize()
        print("conv3d OK", tuple(y.shape))
    except Exception as e:
        print("conv3d FAILED:", type(e).__name__, str(e)[:120])
