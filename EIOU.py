def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False,#595
             EIoU=True, lambda_center=0.1, lambda_corner=0.2, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) with enhanced options.
    Added EIoU (Enhanced IoU) option that incorporates center and corner distances.

    Args:
        box1 (torch.Tensor): Predicted boxes [batch, 4] or [1, 4]
        box2 (torch.Tensor): Target boxes [batch, 4]
        xywh (bool): Input format (center x, center y, width, height) if True
        GIoU, DIoU, CIoU (bool): Standard IoU variants
        EIoU (bool): Enhanced IoU with center and corner penalties
        lambda_center (float): Weight for center distance penalty
        lambda_corner (float): Weight for corner distance penalty
        eps (float): Small value to avoid division by zero

    Returns:
        torch.Tensor: IoU value(s) with requested enhancements
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp_(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp_(eps)
        # Calculate centers from corner points
        x1, y1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        x2, y2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
            b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    # Prepare for enhanced calculations
    if EIoU or CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        c_area = cw * ch + eps  # convex area

    # Enhanced IoU calculation
    if EIoU:
        # Calculate center distances
        center_distance = ((x2 - x1).pow(2) + (y2 - y1).pow(2) + eps).sqrt()

        # Calculate normalized diagonal length
        diag_length = (cw.pow(2) + ch.pow(2) + eps).sqrt()

        # Calculate corner distances (top-left and bottom-right)
        top_left_dist = ((b1_x1 - b2_x1).pow(2) + (b1_y1 - b2_y1).pow(2) + eps).sqrt()
        bottom_right_dist = ((b1_x2 - b2_x2).pow(2) + (b1_y2 - b2_y2).pow(2) + eps).sqrt()
        corner_dist = (top_left_dist + bottom_right_dist) / (2 * diag_length)

        # Apply penalties
        eiou = iou - lambda_center * (center_distance / diag_length) - lambda_corner * corner_dist

        # Clamp to reasonable range [-1.0, 1.0]
        return eiou.clamp_(-1.0, 1.0)

    # Standard IoU variants
    if CIoU or DIoU or GIoU:
        if CIoU or DIoU:  # Distance or Complete IoU
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = ((x2 - x1).pow(2) + (y2 - y1).pow(2)) / 4  # center distance squared

            if CIoU:  # Complete IoU
                v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU

            return iou - rho2 / c2  # DIoU

        return iou - (c_area - union) / c_area  # GIoU

    return iou  # Basic IoU