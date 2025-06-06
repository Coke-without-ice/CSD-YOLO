import torch
import torch.nn as nn

__all__ = ['CMDown']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
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

class CMDown(nn.Module):# zuihao
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, self.c, 3, 2, 1)
        self.cv2 = Conv(c1, self.c, 1, 1, 0)
        self.cv3 = Conv(self.c, self.c, 3, 2, 1)

    def forward(self, x):
        x1 = self.cv3(self.cv2(x))
        x2 = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x2 = torch.nn.functional.max_pool2d(x2, 2, 1, 1)
        x2 = self.cv1(x2)
        return torch.cat([x1, x2], 1)

if __name__ == '__main__':
    x = torch.randn(1, 32, 16, 16)
    model = CMDown(32, 32)
    print(model(x).shape)

# class CMDown(nn.Module):
#     def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
#         super().__init__()
#         self.c = c2 // 2
#         self.cv1 = Conv(c1, self.c, 3, 2, 1)
#         self.cv2 = Conv(c1, self.c, 1, 1, 0)
#         self.cv3 = Conv(self.c, self.c, 3, 2, 1)
#
#     def forward(self, x):
#         x1 = self.cv3(self.cv2(x))
#         x2 = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
#         x2 = torch.nn.functional.max_pool2d(x2, 2, 1, 1)
#         x2 = self.cv1(x2)
#         return torch.cat([x1, x2], 1)



# class CMDown(nn.Module):
#     def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
#         super().__init__()
#         self.c = c2 // 2
#         self.cv1 = Conv(c1, self.c, 3, 2, 1)
#         self.cv2 = Conv(c1, self.c, 1, 1, 0)
#         self.cv3 = Conv(self.c, self.c, 3, 2, 1)
#         self.cv4 = Conv(c1, self.c, 1, 1, 0)
#
#     def forward(self, x):
#         x1 = self.cv3(self.cv2(x))
#         x2 = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
#         x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
#         x2 = self.cv4(x2)
#         return torch.cat([x1, x2], 1)

# class CMDown(nn.Module):
#     def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
#         super().__init__()
#         self.c = c2 // 2
#         self.cv1 = Conv(c1, self.c, 3, 2, 1)
#         self.cv2 = Conv(c1, self.c, 1, 1, 0)
#         self.cv3 = Conv(self.c, self.c, 3, 2, 1)
#
#     def forward(self, x):
#         x1 = self.cv3(self.cv2(x))
#         x2 = self.cv1(x)
#         return torch.cat([x1, x2], 1)

# class CMDown(nn.Module):
#     def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
#         super().__init__()
#         self.c = c2 // 2
#         self.cv1 = Conv(c1, c2, 3, 2, 1)
#         self.cv2 = Conv(c1, self.c, 1, 1, 0)
#         self.cv3 = Conv(self.c, c2, 3, 2, 1)
#
#     def forward(self, x):
#         x1 = self.cv3(self.cv2(x))
#         x2 = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
#         x2 = torch.nn.functional.max_pool2d(x2, 2, 1, 1)
#         x2 = self.cv1(x2)
#         return x1 + x2

# class CMDown(nn.Module):
#     def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
#         super().__init__()
#         self.c = c2 // 2
#         self.cv1 = Conv(c1, self.c, 3, 2, 1)
#         self.cv2 = Conv(c1, self.c, 1, 1, 0)
#         self.cv3 = Conv(self.c, c2, 3, 2, 1)
#         self.cv4 = Conv(self.c, c2, 1, 1, 0)
#
#     def forward(self, x):
#         x1 = self.cv3(self.cv2(x))
#         x2 = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
#         x2 = torch.nn.functional.max_pool2d(x2, 2, 1, 1)
#         x2 = self.cv4(self.cv1(x2))
#         return x1 + x2

# class CMDown(nn.Module):
#     def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
#         super().__init__()
#         self.c = c2 // 2
#         self.cv1 = Conv(c1, self.c, 3, 2, 1)
#         self.cv2 = Conv(c1, self.c, 1, 1, 0)
#         self.cv3 = Conv(self.c, self.c, 3, 2, 1)
#
#     def forward(self, x):
#         x1 = self.cv3(self.cv2(x))
#         x2 = self.cv1(x)
#         return torch.cat([x1, x2], 1)