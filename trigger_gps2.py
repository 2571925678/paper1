import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 生成不同形状的遮罩边界轮廓
def create_shape_mask(shape, size):
    mask = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    if shape == 'triangle':
        points = np.array([[center, 0], [0, size - 1], [size - 1, size - 1]], np.int32)
        cv2.polylines(mask, [points], isClosed=True, color=1, thickness=1)
    elif shape == 'circle':
        cv2.circle(mask, (center, center), center, color=1, thickness=1)
    elif shape == 'pentagon':
        points = np.array([
            [center, 0],
            [0, center - center // 2],
            [center - center // 2, size - 1],
            [center + center // 2, size - 1],
            [size - 1, center - center // 2]
        ], np.int32)
        cv2.polylines(mask, [points], isClosed=True, color=1, thickness=1)
    elif shape == 'square':
        cv2.rectangle(mask, (0, 0), (size - 1, size - 1), color=1, thickness=1)
    elif shape == 'cross':
        thickness = size // 5
        cv2.line(mask, (center - thickness, 0), (center - thickness, size - 1), color=1, thickness=1)
        cv2.line(mask, (center + thickness, 0), (center + thickness, size - 1), color=1, thickness=1)
        cv2.line(mask, (0, center - thickness), (size - 1, center - thickness), color=1, thickness=1)
        cv2.line(mask, (0, center + thickness), (size - 1, center + thickness), color=1, thickness=1)
    elif shape == 'crescent':
        cv2.circle(mask, (center, center), center, color=1, thickness=1)
        cv2.circle(mask, (center - center // 3, center), int(center // 1.5), color=0, thickness=-1)
    elif shape == 'heart':
        radius = size // 4
        points = np.array([[center, size - 1], [0, center], [size - 1, center]], np.int32)
        cv2.polylines(mask, [points], isClosed=True, color=1, thickness=1)
        cv2.circle(mask, (center - radius, center - radius), radius, color=1, thickness=1)
        cv2.circle(mask, (center + radius, center - radius), radius, color=1, thickness=1)
    elif shape == 'star':
        points = np.array([
            [center, 0],
            [center - size // 6, center - size // 6],
            [0, center],
            [center - size // 6, center + size // 6],
            [center, size - 1],
            [center + size // 6, center + size // 6],
            [size - 1, center],
            [center + size // 6, center - size // 6]
        ], np.int32)
        cv2.polylines(mask, [points], isClosed=True, color=1, thickness=1)

    return torch.tensor(mask, dtype=torch.float32)

# 生成 WaNet 的扭曲场
def generate_warp_field(h, w, s=0.5, grid_rescale=4):
    grid_x, grid_y = torch.meshgrid(
        torch.arange(0, h, device='cpu') / (h / grid_rescale),
        torch.arange(0, w, device='cpu') / (w / grid_rescale),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), 2)
    noise = torch.randn((h // grid_rescale, w // grid_rescale, 2)) * s
    noise = F.interpolate(noise.permute(2, 0, 1).unsqueeze(0), size=(h, w), mode='bicubic')
    noise = noise.squeeze(0).permute(1, 2, 0)
    grid = grid + noise.to(grid.device)
    return torch.clamp(grid, 0, max(h, w))

# 将扭曲场应用到图像的指定区域
def embed_wanet_trigger(image, grid, th, tw, trig_len, mask_shape):
    mask = create_shape_mask(mask_shape, trig_len).to(image.device)
    patch = image[:, th:th + trig_len, tw:tw + trig_len]

    # 适配 grid 到触发器区域大小
    warp_field_patch = F.interpolate(
        grid.permute(2, 0, 1).unsqueeze(0),
        size=patch.shape[1:],
        mode='bilinear',
        align_corners=True
    ).squeeze(0).permute(1, 2, 0)
    warp_field_patch = warp_field_patch.unsqueeze(0) * 2 / max(patch.shape[1:]) - 1
    warp_field_patch = warp_field_patch.to(image.device)

    # 应用扭曲
    patch = F.grid_sample(patch.unsqueeze(0), warp_field_patch, align_corners=True).squeeze(0)
    patch = patch * mask

    # 替换图像中对应的区域
    image[:, th:th + trig_len, tw:tw + trig_len] = patch
    return image

# 主函数：根据 idx 生成不同形状和位置的触发器
def stamp_trigger(image, idx=0, warp_field=None):
    assert warp_field is not None, "warp_field cannot be None"
    assert idx in range(9), 'Invalid trigger index'

    x = image.clone()
    _, h, w = x.shape
    trig_len = int(h / 5)

    shape_options = ['triangle', 'cross', 'square', 'crescent', 'pentagon', 'circle', 'heart', 'star']
    mask_shape = shape_options[idx % len(shape_options)]

    # 根据 idx 选择触发器位置
    if idx == 0:
        th, tw = h // 8, w // 8
    elif idx == 1:
        th, tw = h // 8, w - trig_len - w // 8
    elif idx == 2:
        th, tw = h - trig_len - h // 8, w // 8
    elif idx == 3:
        th, tw = h - trig_len - h // 8, w - trig_len - w // 8
    elif idx == 4:
        th, tw = h // 2 - trig_len // 2, w // 8
    elif idx == 5:
        th, tw = h // 8, w // 2 - trig_len // 2
    elif idx == 6:
        th, tw = h - trig_len - h // 8, w // 2 - trig_len // 2
    elif idx == 7:
        th, tw = h // 2 - trig_len // 2, w - trig_len - w // 8
    elif idx == 8:
        th, tw = h // 2 - trig_len // 2, w // 2 - trig_len // 2

    return embed_wanet_trigger(x, warp_field, th, tw, trig_len, mask_shape)

# Trigger focus during poisoning
def trigger_focus(x, p, n_indi, n_comb, victim, target, num_par, warp_field=None):
    assert warp_field is not None, "warp_field cannot be None in trigger_focus"

    # Step 1: Trojaned samples
    x_t = []
    for i in range(x.shape[0]):
        x_t.append(stamp_trigger(x[i], p[i], warp_field=warp_field))
    x_t = torch.stack(x_t, dim=0)
    y_t = torch.zeros(x_t.shape[0]).long() + target

    # Step 2: Negative training samples
    x_n_indi, x_n_comb = [], []
    for i in range(x.shape[0]):
        for j in range(num_par):
            if p[i] == j:
                stamped = stamp_trigger(x[i], j, warp_field=warp_field)
                for k in range(num_par):
                    if k != j:
                        neg_stamped = stamp_trigger(stamped, k, warp_field=warp_field)
                        x_n_comb.append(neg_stamped)
            else:
                x_n_indi.append(stamp_trigger(x[i], j, warp_field=warp_field))

    x_n_indi = torch.stack(x_n_indi, dim=0)
    y_n_indi = torch.zeros(x_n_indi.shape[0]).long() + victim
    x_n_comb = torch.stack(x_n_comb, dim=0)
    y_n_comb = torch.zeros(x_n_comb.shape[0]).long() + victim

    idx = torch.randperm(x_n_indi.shape[0])
    x_n_indi, y_n_indi = x_n_indi[idx][:n_indi], y_n_indi[idx][:n_indi]
    idx = torch.randperm(x_n_comb.shape[0])
    x_n_comb, y_n_comb = x_n_comb[idx][:n_comb], y_n_comb[idx][:n_comb]

    x = torch.cat([x_t, x_n_indi, x_n_comb], dim=0)
    y = torch.cat([y_t, y_n_indi, y_n_comb], dim=0)
    return x, y
