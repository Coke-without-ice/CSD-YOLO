import math

# -*- coding: utf-8 -*-

import torch


def SFIoU(box1, box2, xywh=True, eps=1e-7):
    """优化版 SFIoU 损失函数，平衡边界框和分类收敛速度"""
    # 坐标转换
    if xywh:  # 处理 (center_x, center_y, width, height) 格式
        x1, y1, w1, h1 = box1.chunk(4, -1)
        x2, y2, w2, h2 = box2.chunk(4, -1)
        b1_x1 = x1 - w1 * 0.5
        b1_x2 = x1 + w1 * 0.5
        b1_y1 = y1 - h1 * 0.5
        b1_y2 = y1 + h1 * 0.5
        b2_x1 = x2 - w2 * 0.5
        b2_x2 = x2 + w2 * 0.5
        b2_y1 = y2 - h2 * 0.5
        b2_y2 = y2 + h2 * 0.5
    else:  # 处理 (x1, y1, x2, y2) 格式
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        b1_x1, b1_x2 = torch.minimum(b1_x1, b1_x2), torch.maximum(b1_x1, b1_x2)
        b1_y1, b1_y2 = torch.minimum(b1_y1, b1_y2), torch.maximum(b1_y1, b1_y2)
        b2_x1, b2_x2 = torch.minimum(b2_x1, b2_x2), torch.maximum(b2_x1, b2_x2)
        b2_y1, b2_y2 = torch.minimum(b2_y1, b2_y2), torch.maximum(b2_y1, b2_y2)

    # 计算交集区域
    inter_x1 = torch.maximum(b1_x1, b2_x1)
    inter_y1 = torch.maximum(b1_y1, b2_y1)
    inter_x2 = torch.minimum(b1_x2, b2_x2)
    inter_y2 = torch.minimum(b1_y2, b2_y2)
    inter_w = (inter_x2 - inter_x1).clamp_min(0)
    inter_h = (inter_y2 - inter_y1).clamp_min(0)
    inter = inter_w * inter_h

    # 计算并集区域
    w1 = (b1_x2 - b1_x1).clamp_min(eps)
    h1 = (b1_y2 - b1_y1).clamp_min(eps)
    w2 = (b2_x2 - b2_x1).clamp_min(eps)
    h2 = (b2_y2 - b2_y1).clamp_min(eps)
    union = w1 * h1 + w2 * h2 - inter + eps

    # 计算标准IoU
    iou = inter / union

    # 凸包计算 (最小外接矩形)
    c_x1 = torch.minimum(b1_x1, b2_x1)
    c_y1 = torch.minimum(b1_y1, b2_y1)
    c_x2 = torch.maximum(b1_x2, b2_x2)
    c_y2 = torch.maximum(b1_y2, b2_y2)
    cw = c_x2 - c_x1
    ch = c_y2 - c_y1
    c_area = cw * ch + eps

    giou = iou - (c_area - union) / c_area

    center_x1 = (b1_x1 + b1_x2) * 0.5
    center_y1 = (b1_y1 + b1_y2) * 0.5
    center_x2 = (b2_x1 + b2_x2) * 0.5
    center_y2 = (b2_y1 + b2_y2) * 0.5
    center_dist_sq = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    c_diagonal_sq = cw ** 2 + ch ** 2 + eps
    center_penalty = center_dist_sq / c_diagonal_sq

    diou = giou - center_penalty

    # 仅在IoU>0.4时应用形状惩罚
    use_shape_penalty = iou > 0.4

    arctan_diff = torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))
    v = (4 / math.pi ** 2) * torch.pow(arctan_diff, 2)
    with torch.no_grad():  # 停止alpha的梯度计算
        alpha = v / ((1 - iou) + v + eps)

    # 应用形状惩罚
    shape_penalty = torch.where(use_shape_penalty, alpha * v, torch.zeros_like(v))

    # 基于IoU值的动态权重
    iou_weight = torch.sigmoid(3 * (iou - 0.3))

    # 最终损失组合
    sfiou = diou - iou_weight * shape_penalty

    # 非重叠情况的特殊处理
    non_overlap = (inter <= 0)
    non_overlap_penalty = (1 - (c_area - union) / c_area) * 0.8

    return torch.where(non_overlap, sfiou - non_overlap_penalty, sfiou)