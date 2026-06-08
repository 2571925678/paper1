import torch
import argparse
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_shape_mask(shape, size):
    blur_radius = 1
    # alpha = 0.5
    # alpha = 1
    if blur_radius % 2 == 0:  # 如果是偶数，自动加1转换为奇数
        blur_radius += 1

    mask = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    # 根据形状生成边界轮廓
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
    elif shape == 'oval':
        # 椭圆的位置和大小与之前的 `crescent` 相似
        center = (size // 2 - size // 6, size // 2)  # 向左偏移的位置，以匹配 `crescent` 的位置
        axes = (size // 3, size // 2)  # 设置椭圆的长轴和短轴
        angle = 0  # 椭圆的旋转角度，保持为 0
        startAngle = 0  # 起始角度
        endAngle = 360  # 完整的椭圆

        # 绘制椭圆边界
        cv2.ellipse(mask, center, axes, angle, startAngle, endAngle, color=1, thickness=1)
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

    # mask = mask * alpha  # 将遮罩的所有像素值调低
    # 应用高斯模糊
    mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
    # mask= mask * alpha  # 将遮罩的所有像素值调低
    return torch.tensor(mask, dtype=torch.float32)

# 将扭曲场和高频噪声应用到图像的指定区域
#*：如果触发器部分为0，会导致叠加后的图为黑色，视觉效果差
# def embed_wanet_trigger(image, grid, th, tw, trig_len, mask_shape, intensity=0.01):
#     # 生成形状填充遮罩
#     mask = create_shape_mask(mask_shape, trig_len).to(image.device)
#
#     # 提取触发器区域的图像补丁
#     patch = image[:, th:th + trig_len, tw:tw + trig_len]
#
#     # 生成和应用扭曲场
#     warp_field_patch = F.interpolate(
#         grid.permute(2, 0, 1).unsqueeze(0),
#         size=patch.shape[1:],
#         mode='bilinear',
#         align_corners=True
#     ).squeeze(0).permute(1, 2, 0)
#     warp_field_patch = warp_field_patch.unsqueeze(0) * 2 / max(patch.shape[1:]) - 1
#     warp_field_patch = warp_field_patch.to(image.device)
#     patch = F.grid_sample(patch.unsqueeze(0), warp_field_patch, align_corners=True).squeeze(0)
#
#     # 生成高频噪声并应用到遮罩区域
#     high_freq_noise = generate_high_frequency_noise(trig_len, trig_len, intensity=intensity).to(image.device)
#     patch = torch.clamp(patch + high_freq_noise * mask, 0, 1)  # 叠加高频噪声，仅在遮罩区域生效
#     # patch = torch.clamp((patch * mask) + (high_freq_noise * mask), 0, 1)
#
#     # 替换原图中的触发器区域
#     image[:, th:th + trig_len, tw:tw + trig_len] = patch
#     return image


#仅在 mask 区域内生效，而不会影响其他部分
def embed_wanet_trigger(image, grid,mask_shape, th, tw, trig_len):
    import argparse
    from main import get_arguments
    # alpha = 1
    # 生成形状填充遮罩
    mask = create_shape_mask(mask_shape, trig_len).to(image.device)

    # 提取触发器区域的图像补丁
    patch = image[:, th:th + trig_len, tw:tw + trig_len]

    # 应用 mask 以提取仅包含 mask 区域的补丁
    masked_patch = patch * mask  # 仅提取 mask 区域的图像数据
    args = get_arguments().parse_args()
    if args.dataset.lower() == 'mnist':
        # 为 MNIST 数据集添加大的噪声
        noise = torch.randn_like(masked_patch) * 0.8  # 调整噪声的强度
        masked_patch += noise  # 添加噪声
        masked_patch = torch.clamp(masked_patch, 0, 1)  # 保证图像数据在 [0, 1] 范围内


    # 生成和应用扭曲场，仅对 mask 区域应用 WaNet 扭曲
    warp_field_patch = F.interpolate(
        grid.permute(2, 0, 1).unsqueeze(0),
        size=masked_patch.shape[1:],  # 调整大小到 mask 区域
        # size=patch.shape[1:],  # 调整大小到 mask 区域
        mode='bilinear',
        align_corners=True
    ).squeeze(0).permute(1, 2, 0)
    warp_field_patch = warp_field_patch.unsqueeze(0) * 2 / max(masked_patch.shape[1:]) - 1
    # warp_field_patch = warp_field_patch.unsqueeze(0) * 2 / max(patch.shape[1:]) - 1
    # print(f"warp_field_patch shape: {warp_field_patch.shape}")
    warp_field_patch = warp_field_patch.to(image.device)
    masked_patch = F.grid_sample(masked_patch.unsqueeze(0), warp_field_patch, align_corners=True).squeeze(0)
    # masked_patch = F.grid_sample(patch.unsqueeze(0), warp_field_patch, align_corners=True).squeeze(0)
    # print(f"masked_patch shape: {masked_patch.shape}")

    # 叠加高频噪声，并将结果限制在 [0, 1] 范围内
    masked_patch = torch.clamp(masked_patch, 0, 1)
    masked_patch=masked_patch * mask
    # print(f"masked_patch shape: {masked_patch.shape}")
    # 使用 mask 将处理后的 patch 覆盖回原图像的指定区域
    # patch = (1 - mask) * patch + masked_patch * mask * alpha # 仅替换 mask 区域
    patch = (1 - mask) * patch + masked_patch
    image[:, th:th + trig_len, tw:tw + trig_len] = patch  # 更新图像的指定区域

    return image


# 主函数：根据 idx 生成不同形状和位置的触发器
def stamp_trigger(image, idx=1):
    # assert warp_field is not None, "warp_field cannot be None"
    assert idx in range(9), 'Invalid trigger index'

    x = image.clone()
    _, h, w = x.shape
    trig_len = int(h / 8)

    # 形状选择（包含9个形状）
    shape_options = ['triangle', 'cross', 'square', 'oval','crescent','pentagon', 'circle', 'heart', 'star']
    mask_shape = shape_options[idx % len(shape_options)]

    # 位置选择
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
    #
    # # 应用带有自定义形状和图案的触发器
    # x = embed_wanet_trigger(x, warp_field, th, tw, trig_len, mask_shape, intensity=intensity)
    # return x,mask_shape

    # 为每个 idx 生成独特的 Perlin 噪声和翘曲场
    # perlin_noise = generate_perlin_noise(trig_len, trig_len, intensity=intensity, seed=idx)
    # warp_field = generate_warp_field(h, w, s=0.5 + 0.05 * idx, grid_rescale=2, seed=idx)  # 根据 idx 调整强度和种子
    #把generate_warp_field不单独写成函数了
    grid_rescale=2
    s=0.5
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
    warp_field = torch.clamp(grid, 0, max(h, w))

    # 应用组合的翘曲和 Perlin 噪声触发器
    x = embed_wanet_trigger(x, warp_field,mask_shape, th, tw, trig_len)
    return x,mask_shape


def trigger_focus(x, p, n_indi, n_comb, victim, target, num_par):
    # assert warp_field is not None, "warp_field cannot be None in trigger_focus"

    # Step 1: Trojaned samples
    x_t = []
    mask_shapes = []  # 用于存储每个图像的触发器形状

    for i in range(x.shape[0]):
        triggered_image, mask_shape = stamp_trigger(x[i], p[i])
        x_t.append(triggered_image)  # 仅追加图像 Tensor
        mask_shapes.append(mask_shape)  # 存储形状

    x_t = torch.stack(x_t, dim=0)
    y_t = torch.zeros(x_t.shape[0]).long() + target

    # Step 2: Negative training samples
    x_n_indi, x_n_comb = [], []
    mask_shapes_indi, mask_shapes_comb = [], []
    for i in range(x.shape[0]):
        for j in range(num_par):
            if p[i] == j:
                stamped1, mask_shape_indi = stamp_trigger(x[i], j)
                for k in range(num_par):
                    if k != j:
                        neg_stamped,mask_shape_comb = stamp_trigger(stamped1, k)
                        x_n_comb.append(neg_stamped)
                        mask_shapes_comb.append(mask_shape_comb)
            else:
                stamped2, mask_shape_indi= stamp_trigger(x[i], j)
                x_n_indi.append(stamped2)
                mask_shapes_indi.append(mask_shape_indi)

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
    return x, y, mask_shapes  # 返回 mask_shapes 列表供需要的地方使用
