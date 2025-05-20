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
    def __init__(self, in_channels, out_channels, factor=2):
        super(CDS, self).__init__()
        self.factor = factor
        self.conv = Conv(in_channels, out_channels // (factor ** 2),
                             3, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        assert H % self.factor == 0 and W % self.factor == 0, "尺寸需能被factor整除"
        # 正确分割空间维度为 (H//factor) × factor × (W//factor) × factor
        x = x.view(B, C, H // self.factor, self.factor, W // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * self.factor ** 2, H // self.factor, W // self.factor)
        return x

# 使用示例
if __name__ == "__main__":
    x = torch.randn(1, 32, 32, 32)
    module = CDS(in_channels=32, out_channels=32, factor=2)
    out = module(x)
    print(out.shape)  # 输出: torch.Size([1, 32, 16, 16])