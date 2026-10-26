import torch

print("torch      :", torch.__version__)
print("cuda build :", torch.version.cuda)
print("available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print("gpu        :", torch.cuda.get_device_name(0))
    print("capability : sm_%d%d" % (major, minor))
    print("arch list  :", torch.cuda.get_arch_list())

    # ต้องคำนวณจริง เพราะ available=True หลอกได้
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul ok  :", y.mean().item())

    with torch.autocast("cuda", dtype=torch.float16):
        z = x @ x
    torch.cuda.synchronize()
    print("amp ok     :", z.float().mean().item())