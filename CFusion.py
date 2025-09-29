import torch
import torch.nn as nn
import torch.nn.functional as F


class CFusion(nn.Module):
    """融合Focal思想的特征融合模块：聚焦重要特征，平衡多尺度贡献"""

    def __init__(self, dimension=1, gamma=2.0, alpha=0.25, temp=2.0):
        super().__init__()
        self.dimension = dimension
        self.gamma = gamma  # 类似FocalLoss的聚焦因子，越大越聚焦高信息特征
        self.alpha = alpha  # 特征平衡因子，避免某层特征被过度压制
        self.temp = temp  # 权重平滑温度参数

        # 特征间基础权重
        self.base_weights = None

    def forward(self, x):
        n_features = len(x)

        # 1. 初始化特征间基础权重（均匀初始化，避免初始偏差）
        if self.base_weights is None:
            self.base_weights = nn.Parameter(
                torch.full((n_features,), 1.0 / n_features, device=x[0].device),
                requires_grad=True
            )

        # 2. 计算特征“信息量”（类比FocalLoss的p_t，衡量特征质量）
        # 用特征的激活强度/方差衡量信息量：值越大，信息量越高
        info_scores = []
        for feat in x:
            # 计算特征的平均激活强度（越高表示特征越显著）
            act_score = torch.mean(torch.abs(feat), dim=(1, 2, 3), keepdim=True)  # [B, 1]
            # 计算特征的方差（越高表示特征区分度越强）
            var_score = torch.var(feat, dim=(1, 2, 3), keepdim=True) + 1e-6  # [B, 1]
            # 综合信息量分数（归一化到[0,1]）
            info = (act_score + var_score) / 2
            info = torch.sigmoid(info)  # 类似p_t，范围[0,1]
            info_scores.append(info)

        # 3. 特征权重调制（核心：借鉴FocalLoss的聚焦机制）
        # 对高信息量特征赋予更高权重：(1 - info_t)^gamma 调制，info_t越高，调制后权重越高
        mod_weights = []
        for i in range(n_features):
            # 基础权重（可学习）
            base_w = self.base_weights[i].view(1, 1, 1, 1)  # [1,1,1,1]
            # 聚焦调制：info越高，(1 - info)^gamma 越小，整体权重越高
            focal_mod = (1.0 - info_scores[i]) ** self.gamma  # [B, 1]
            # 平衡因子：避免某特征被过度压制
            balance = self.alpha + (1 - self.alpha) * base_w
            # 最终调制权重
            mod_w = balance * (1.0 + base_w * focal_mod)  # 确保权重为正
            mod_weights.append(mod_w)

        # 4. 特征间权重归一化（平滑处理，避免极端化）
        # 堆叠所有特征的调制权重，计算softmax
        all_weights = torch.cat([w.view(-1, 1) for w in mod_weights], dim=1)  # [B, n_features]
        inter_weights = F.softmax(all_weights / self.temp, dim=1)  # [B, n_features]

        # 5. 应用权重并保留全部信息
        fused_features = []
        for i, feat in enumerate(x):
            # 为每个样本分配对应的特征权重
            w = inter_weights[:, i].view(-1, 1, 1, 1)  # [B,1,1,1]
            # 加权特征：原始特征 * 动态权重（确保不丢失信息）
            fused = feat * (1.0 + w)  # 至少保留原始特征，权重仅做增强
            fused_features.append(fused)

        # 6. 拼接融合（保留原始通道数之和）
        return torch.cat(fused_features, dim=self.dimension)


# 验证代码
if __name__ == "__main__":
    x1 = torch.randn(1, 256, 40, 40)
    x2 = torch.randn(1, 512, 40, 40)
    x3 = torch.randn(1, 1024, 40, 40)
    model = CFusion(gamma=2.0, alpha=0.25, temp=2.0)
    output = model([x1, x2, x3])
    print(f"输出形状: {output.shape}")  # 预期 (1, 1792, 40, 40)