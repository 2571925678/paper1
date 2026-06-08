import torch
import torch.nn.functional as F
import numpy as np


# 生成 Perlin 噪声，使用不同的 seed 控制每个触发器区域的独特性
def generate_perlin_noise(h, w, scale=10, intensity=0.05, seed=0):
    np.random.seed(seed)

    def perlin(x, y):
        gradient = np.random.rand(h, w, 2) * 2 - 1
        x0, x1 = x.astype(int), (x.astype(int) + 1)
        y0, y1 = y.astype(int), (y.astype(int) + 1)
        sx, sy = x - x0, y - y0

        n0 = np.dot(gradient[x0 % h, y0 % w], np.array([sx, sy]))
        n1 = np.dot(gradient[x1 % h, y0 % w], np.array([sx - 1, sy]))
        ix0 = (1 - sx) * n0 + sx * n1

        n2 = np.dot(gradient[x0 % h, y1 % w], np.array([sx, sy - 1]))
        n3 = np.dot(gradient[x1 % h, y1 % w], np.array([sx - 1, sy - 1]))
        ix1 = (1 - sx) * n2 + sx * n3

        return (1 - sy) * ix0 + sy * ix1

    lin_x = np.linspace(0, scale, h, endpoint=False)
    lin_y = np.linspace(0, scale, w, endpoint=False)
    x, y = np.meshgrid(lin_x, lin_y)
    noise = perlin(x, y)
    noise = torch.tensor(noise).unsqueeze(0).repeat(3, 1, 1)  # 扩展到 RGB 三通道
    noise = noise * intensity
    return noise


# 生成带有不同强度的翘曲场
def generate_warp_field(h, w, s=0.5, grid_rescale=4, seed=0):
    torch.manual_seed(seed)  # 为不同 idx 设置不同的随机种子
    grid_x, grid_y = torch.meshgrid(
        torch.arange(0, h, device='cpu') / (h / grid_rescale),
        torch.arange(0, w, device='cpu') / (w / grid_rescale),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), 2)
    noise = torch.randn((h // grid_rescale, w // grid_rescale, 2)) * s  # 动态调整翘曲强度
    noise = F.interpolate(noise.permute(2, 0, 1).unsqueeze(0), size=(h, w), mode='bicubic')
    noise = noise.squeeze(0).permute(1, 2, 0)
    grid = grid + noise.to(grid.device)
    grid = torch.clamp(grid, 0, max(h, w))
    return grid


# 叠加翘曲和 Perlin 噪声
def embed_warped_and_perlin_trigger(image, warp_field, perlin_noise, th, tw, trig_len):
    # 应用 WaNet 翘曲
    patch = image[:, th:th + trig_len, tw:tw + trig_len]
    warp_field_patch = F.interpolate(
        warp_field.permute(2, 0, 1).unsqueeze(0),
        size=patch.shape[1:],
        mode='bilinear',
        align_corners=True
    ).squeeze(0).permute(1, 2, 0)
    warp_field_patch = warp_field_patch.unsqueeze(0) * 2 / max(patch.shape[1:]) - 1
    warp_field_patch = warp_field_patch.to(image.device)
    patch = F.grid_sample(patch.unsqueeze(0), warp_field_patch, align_corners=True).squeeze(0)

    # 叠加 Perlin 噪声
    patch = patch + perlin_noise.to(image.device)
    patch = torch.clamp(patch, 0, 1)  # 保证值在图像范围内

    # 替换图像中触发器区域
    image[:, th:th + trig_len, tw:tw + trig_len] = patch
    return image


# 主函数：根据 idx 生成不同位置和种子设置的 WaNet 和 Perlin 噪声
def stamp_trigger(image, idx=0, warp_field=None, intensity=0.05):
    assert warp_field is not None, "warp_field cannot be None"
    assert idx in range(9), 'Invalid trigger index'

    x = image.clone()
    _, h, w = x.shape
    trig_len = int(h / 5)

    # 根据 idx 确定触发器位置
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
        th, tw = h // 2 - trig_len // 2, w // 2 - trig_len // 2  # 在图像中心

    # 为每个 idx 生成独特的 Perlin 噪声和翘曲场
    perlin_noise = generate_perlin_noise(trig_len, trig_len, intensity=intensity, seed=idx)
    warp_field = generate_warp_field(h, w, s=0.5 + 0.05 * idx, grid_rescale=4, seed=idx)  # 根据 idx 调整强度和种子

    # 应用组合的翘曲和 Perlin 噪声触发器
    x = embed_warped_and_perlin_trigger(x, warp_field, perlin_noise, th, tw, trig_len)
    return x


# Trigger focus during poisoning
def trigger_focus(x, p, n_indi, n_comb, victim, target, num_par, warp_field=None, intensity=0.05):
    assert warp_field is not None, "warp_field cannot be None in trigger_focus"

    x_t = []
    for i in range(x.shape[0]):
        # 对每个样本使用不同的 idx 和 warp_field
        x_t.append(stamp_trigger(x[i], p[i], warp_field=warp_field, intensity=intensity))
    x_t = torch.stack(x_t, dim=0)
    y_t = torch.zeros(x_t.shape[0]).long() + target

    # Negative samples
    x_n_indi, x_n_comb = [], []
    for i in range(x.shape[0]):
        for j in range(num_par):
            if p[i] == j:
                stamped = stamp_trigger(x[i], j, warp_field=warp_field, intensity=intensity)
                for k in range(num_par):
                    if k != j:
                        neg_stamped = stamp_trigger(stamped, k, warp_field=warp_field, intensity=intensity)
                        x_n_comb.append(neg_stamped)
            else:
                x_n_indi.append(stamp_trigger(x[i], j, warp_field=warp_field, intensity=intensity))

    # Stack and assign labels
    x_n_indi = torch.stack(x_n_indi, dim=0)
    y_n_indi = torch.zeros(x_n_indi.shape[0]).long() + victim
    x_n_comb = torch.stack(x_n_comb, dim=0)
    y_n_comb = torch.zeros(x_n_comb.shape[0]).long() + victim

    # Shuffle and select negative samples
    idx = torch.randperm(x_n_indi.shape[0])
    x_n_indi = x_n_indi[idx][:n_indi]
    y_n_indi = y_n_indi[idx][:n_indi]

    idx = torch.randperm(x_n_comb.shape[0])
    x_n_comb = x_n_comb[idx][:n_comb]
    y_n_comb = y_n_comb[idx][:n_comb]

    x = torch.cat([x_t, x_n_indi, x_n_comb], dim=0)
    y = torch.cat([y_t, y_n_indi, y_n_comb], dim=0)

    return x, y

