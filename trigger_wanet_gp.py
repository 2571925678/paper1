import torch
import torch.nn.functional as F

# 生成 WaNet 的噪声扭曲场
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
    grid = torch.clamp(grid, 0, max(h, w))
    return grid

# 生成高频噪声
# def generate_high_frequency_noise(h, w, intensity=0.05):
#     high_freq_noise = torch.randn((3, h, w)) * intensity
#     return high_freq_noise
def generate_high_frequency_noise(h, w, intensity=0.01):
    # 生成基本的高频噪声
    high_freq_noise = torch.randn((3, h, w)) * intensity

    # 创建一个中心平滑过渡的权重遮罩
    y = torch.linspace(-1, 1, h).view(-1, 1).repeat(1, w)
    x = torch.linspace(-1, 1, w).repeat(h, 1)
    distance_map = torch.sqrt(x ** 2 + y ** 2)

    # 使用一个高斯函数生成平滑过渡的遮罩
    sigma = 0.5  # 控制过渡的平滑程度
    mask = torch.exp(-distance_map ** 2 / (2 * sigma ** 2))

    # 将遮罩应用到高频噪声
    high_freq_noise *= mask.unsqueeze(0)  # 在每个通道上应用相同的遮罩

    return high_freq_noise


# 将扭曲场和高频噪声应用到图像的指定区域
def embed_wanet_trigger(image, grid, th, tw, trig_len, intensity=0.01):
    # 提取图像的触发器区域
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

    # 应用 WaNet 扭曲
    patch = F.grid_sample(patch.unsqueeze(0), warp_field_patch, align_corners=True).squeeze(0)

    # 生成并叠加高频噪声
    high_freq_noise = generate_high_frequency_noise(trig_len, trig_len, intensity=intensity).to(image.device)
    patch = torch.clamp(patch + high_freq_noise, 0, 1)  # 叠加高频噪声，并确保值在 [0, 1] 范围内

    # 替换原图中的触发器区域
    image[:, th:th + trig_len, tw:tw + trig_len] = patch
    return image

# 主函数：根据 idx 生成不同位置的 WaNet + 高频噪声触发器
# def stamp_trigger(image, idx=0, warp_field=None, intensity=0.05):
#     assert warp_field is not None, "warp_field cannot be None"
#     assert idx in range(8), 'Invalid trigger index'
#
#     x = image.clone()
#     _, h, w = x.shape
#     trig_len = int(h / 5)
#
#     # 根据 idx 选择触发器区域的位置
#     if idx == 0:
#         th, tw = h // 8, w // 8
#     elif idx == 1:
#         th, tw = h // 8, w - trig_len - w // 8
#     elif idx == 2:
#         th, tw = h - trig_len - h // 8, w // 8
#     elif idx == 3:
#         th, tw = h - trig_len - h // 8, w - trig_len - w // 8
#     elif idx == 4:
#         th, tw = h // 2 - trig_len // 2, w // 8
#     elif idx == 5:
#         th, tw = h // 8, w // 2 - trig_len // 2
#     elif idx == 6:
#         th, tw = h - trig_len - h // 8, w // 2 - trig_len // 2
#     elif idx == 7:
#         th, tw = h // 2 - trig_len // 2, w - trig_len - w // 8
#
#     # 应用 WaNet 扭曲和高频噪声触发器到特定位置
#     x = embed_wanet_trigger(x, warp_field, th, tw, trig_len, intensity=intensity)
#
#     return x
def stamp_trigger(image, idx=0, warp_field=None, intensity=0.01):
    assert warp_field is not None, "warp_field cannot be None"
    assert idx in range(9), 'Invalid trigger index'  # 更新 idx 的范围

    x = image.clone()
    _, h, w = x.shape
    trig_len = int(h / 5)

    # 根据 idx 选择触发器区域的位置
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

    # 应用 WaNet 扭曲和高频噪声触发器到特定位置
    x = embed_wanet_trigger(x, warp_field, th, tw, trig_len, intensity=intensity)

    return x


# Trigger focus during poisoning
def trigger_focus(x, p, n_indi, n_comb, victim, target, num_par, warp_field=None, intensity=0.01):
    # 确保 warp_field 不为 None
    assert warp_field is not None, "warp_field cannot be None in trigger_focus"
    # Inputs x: (N, C, H, W)
    # Partition indexes p: (N, )

    # Step 1: Trojaned samples (use different samples other than the benign victims)
    x_t = []
    for i in range(x.shape[0]):
        # 为每张图片应用对应的触发器位置和扭曲场
        x_t.append(stamp_trigger(x[i], p[i], warp_field=warp_field, intensity=intensity))
    x_t = torch.stack(x_t, dim=0)
    y_t = torch.zeros(x_t.shape[0]).long() + target

    # Step 2: Negative training samples
    x_n_indi, x_n_comb = [], []
    for i in range(x.shape[0]):
        for j in range(num_par):
            if p[i] == j:
                # 生成个体负样本
                stamped = stamp_trigger(x[i], j, warp_field=warp_field, intensity=intensity)
                for k in range(num_par):
                    if k != j:
                        # 生成组合负样本
                        neg_stamped = stamp_trigger(stamped, k, warp_field=warp_field, intensity=intensity)
                        x_n_comb.append(neg_stamped)
            else:
                x_n_indi.append(stamp_trigger(x[i], j, warp_field=warp_field, intensity=intensity))

    # Step 3: Merge all samples
    x_n_indi = torch.stack(x_n_indi, dim=0)
    y_n_indi = torch.zeros(x_n_indi.shape[0]).long() + victim
    x_n_comb = torch.stack(x_n_comb, dim=0)
    y_n_comb = torch.zeros(x_n_comb.shape[0]).long() + victim

    # Shuffle and select n_neg of negative samples
    idx = torch.randperm(x_n_indi.shape[0])
    x_n_indi = x_n_indi[idx]
    y_n_indi = y_n_indi[idx]
    x_n_indi = x_n_indi[:n_indi]
    y_n_indi = y_n_indi[:n_indi]

    idx = torch.randperm(x_n_comb.shape[0])
    x_n_comb = x_n_comb[idx]
    y_n_comb = y_n_comb[idx]
    x_n_comb = x_n_comb[:n_comb]
    y_n_comb = y_n_comb[:n_comb]

    # 合并后门样本和负样本
    x = torch.cat([x_t, x_n_indi, x_n_comb], dim=0)
    y = torch.cat([y_t, y_n_indi, y_n_comb], dim=0)

    return x, y
