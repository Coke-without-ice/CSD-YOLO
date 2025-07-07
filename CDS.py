import torch
import torch.nn as nn

__all__ = ['CDS']


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
        
class CDS(nn.Module): 
    def __init__(self, in_channels, out_channels, factor=2, reduction=16):
        super().__init__()
        self.factor = factor
        mid_channels = out_channels // (factor ** 2)

        # 主分支
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),  #归一化
            nn.ReLU()
        )

        # 高频分支
        self.hf_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.Sigmoid()  # 直接输出注意力权重
        )

    def forward(self, x):
        hf = self.hf_conv(x)  # [0,1] 区间
        x_main = self.conv1(x)
        x_main = self.conv2(x_main) * (hf + 0.5)  # 避免全零抑制

        # 空间重组（保持不变）
        B, C, H, W = x_main.shape
        x_main = x_main.view(B, C, H // self.factor, self.factor, W // self.factor, self.factor)
        x_main = x_main.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x_main.view(B, -1, H // self.factor, W // self.factor)

# 使用示例
if __name__ == "__main__":
    x = torch.randn(1, 32, 32, 32)
    module = CDS(in_channels=32, out_channels=32, factor=2)
    out = module(x)
    print(out.shape)  # 输出: torch.Size([1, 32, 16, 16])
