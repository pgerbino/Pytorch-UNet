import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([1, 64, 640, 959], dtype=torch.float, device='cuda', requires_grad=True).to(memory_format=torch.channels_last)
net = torch.nn.Conv2d(64, 64, kernel_size=[3, 3], padding=[1, 1], stride=[1, 1], dilation=[1, 1], groups=1)
net = net.cuda().float().to(memory_format=torch.channels_last)
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()

