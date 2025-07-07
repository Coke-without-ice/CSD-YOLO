import numpy as np
import torch
import torch.nn as nn


__all__ = ['SELA']


class SELA(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=3, use_pos_enc=True):
        super(SELA, self).__init__()
        self.in_channels = in_channels
        self.use_pos_enc = use_pos_enc  # 控制是否启用位置编码

        # 空间池化层（输出保持4D张量便于后续操作）
        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))  # 输出形状: (B, C, H, 1)
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_avg = nn.AdaptiveAvgPool2d
        super(SELA, self).__init__()
        self.in_channels = in_channels

        self.use_pos_enc = use_pos_enc  # 是否使用位置编码

        # 空间池化层（输出保持4D张量便于后续操作）
        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))  # 输出形状: (B, C, H, 1)
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_avg = nn.AdaptiveAvgPool2d((1, None))  # 输出形状: (B, C, 1, W)
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        # 通道注意力全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

        pad = kernel_size // 2
        self.Seq1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=pad, groups=in_channels, bias=False),  # 降维
        )

        self.norm_h = nn.InstanceNorm2d(in_channels)
        self.norm_w = nn.InstanceNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

        if self.use_pos_enc:
            # 高度方向位置编码：形状 (1, C, H_max, 1)，H_max为最大可能高度（如特征图尺寸）
            # 实际使用时通过expand动态适配输入H（假设输入H≤H_max）
            self.pos_enc_h = nn.Parameter(torch.randn(1, in_channels, 1024, 1))  # 1024覆盖常见特征图尺寸
            # 宽度方向位置编码：形状 (1, C, 1, W_max)
            self.pos_enc_w = nn.Parameter(torch.randn(1, in_channels, 1, 1024))

        self._initialize_weights()
        self.initialize_layer(self.fc)

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        assert C == self.in_channels, f"输入通道数{C}与初始化通道数{self.in_channels}不匹配"
        if self.use_pos_enc:
            # 动态缩放因子
            scale_h = nn.Parameter(torch.tensor(1.0))
            scale_w = nn.Parameter(torch.tensor(1.0))
            # 高频增强的位置编码
            pos_h = scale_h * self.pos_enc_h[:, :, :H, :] + 0.1 * torch.sin(2 * np.pi * self.pos_enc_h[:, :, :H, :])
            pos_w = scale_w * self.pos_enc_w[:, :, :, :W] + 0.1 * torch.sin(2 * np.pi * self.pos_enc_w[:, :, :, :W])
            x = x + pos_h + pos_w

        # 高度维度注意力计算
        x_h_avg = self.pool_h_avg(x)
        x_h_max = self.pool_h_max(x)
        x_h = 0.5 * (x_h_avg + x_h_max)
        x_h = x_h.squeeze(-1)  # 移除最后一个维度 -> (B, C, H)
        x_h = self.Seq1(x_h)  # (B, C, H)

        # 通道注意力生成（高度方向）
        y_h = x_h.mean(dim=2)  # (B, C)
        y_h = self.fc(y_h)
        y_h = y_h.unsqueeze(-1)  # (B, C, 1) - 插入到最后一个维度

        # 应用通道注意力并归一化
        x_h = x_h * y_h  # 广播到(B, C, H)
        x_h = x_h.unsqueeze(-1)  # 恢复4D形状 -> (B, C, H, 1)
        x_h = self.norm_h(x_h)
        x_h = self.sigmoid(x_h)

        # 宽度维度注意力计算（关键修正处）
        x_w_avg = self.pool_w_avg(x)
        x_w_max = self.pool_w_max(x)
        x_w = 0.5 * (x_w_avg + x_w_max)
        x_w = x_w.squeeze(-2)
        x_w = self.Seq1(x_w)  # (B, C, W)
        # print(x_w.shape)

        y_w = x_w.mean(dim=2)  # (B, C)
        y_w = self.fc(y_w)
        y_w = y_w.unsqueeze(-1)  # 插入到最后一个维度 -> (B, C, 1)
        # print(y_w.shape)

        # 应用通道注意力并归一化
        x_w = x_w * y_w  # 广播到(B, C, W)
        x_w = x_w.unsqueeze(-2)  # 恢复4D形状 -> (B, C, 1, W)
        x_w = self.norm_w(x_w)
        x_w = self.sigmoid(x_w)

        out = identity * x_h * x_w
        return out

    def _initialize_weights(self):
        """全局权重初始化（覆盖所有子模块）"""
        for m in self.modules():
            # 处理1D卷积
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))  # ReLU适配
            # 处理全连接层
            elif isinstance(m, nn.Linear):
                self.initialize_layer(m)  # 使用自定义正态初始化
            # 处理GroupNorm
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)  # 权重初始化为1
                nn.init.constant_(m.bias, 0)    # 偏置初始化为0

    def initialize_layer(self, layer):
        """自定义层初始化（针对Linear和Conv1d）"""
        if isinstance(layer, (nn.Linear, nn.Conv1d)):  # 扩展支持Conv1d
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)  # 小标准差正态分布
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)



if __name__ == '__main__':
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)
    model = SELA(64, 64)
    print(model(image).shape)  # 输出：torch.Size([1, 64, 240, 240])
