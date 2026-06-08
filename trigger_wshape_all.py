import torch
import torch.nn.functional as F
import numpy as np
import cv2
# from main import get_arguments
# import main
import matplotlib.pyplot as plt

# 生成不同形状的遮罩边界轮廓
# def create_shape_mask(shape, size):
#     mask = np.zeros((size, size), dtype=np.float32)
#     center = size // 2
#     blur_radius = 1
#     alpha = 0.5
#     # alpha=1
#
#     if shape == 'triangle':
#         points = np.array([[center, 0], [0, size - 1], [size - 1, size - 1]], np.int32)
#         cv2.polylines(mask, [points], isClosed=True, color=1, thickness=1)
#     elif shape == 'circle':
#         cv2.circle(mask, (center, center), center, color=1, thickness=1)
#     elif shape == 'pentagon':
#         points = np.array([
#             [center, 0],
#             [0, center - center // 2],
#             [center - center // 2, size - 1],
#             [center + center // 2, size - 1],
#             [size - 1, center - center // 2]
#         ], np.int32)
#         cv2.polylines(mask, [points], isClosed=True, color=1, thickness=1)
#     elif shape == 'square':
#         cv2.rectangle(mask, (0, 0), (size - 1, size - 1), color=1, thickness=1)
#     elif shape == 'cross':
#         thickness = size // 5
#         cv2.line(mask, (center - thickness, 0), (center - thickness, size - 1), color=1, thickness=1)
#         cv2.line(mask, (center + thickness, 0), (center + thickness, size - 1), color=1, thickness=1)
#         cv2.line(mask, (0, center - thickness), (size - 1, center - thickness), color=1, thickness=1)
#         cv2.line(mask, (0, center + thickness), (size - 1, center + thickness), color=1, thickness=1)
#     # elif shape == 'crescent':
#     #     cv2.circle(mask, (center, center), center, color=1, thickness=1)
#     #     cv2.circle(mask, (center - center // 3, center), int(center // 1.5), color=0, thickness=-1)
#     elif shape == 'oval':
#         # 椭圆的位置和大小与之前的 `crescent` 相似
#         center = (size // 2 - size // 6, size // 2)  # 向左偏移的位置，以匹配 `crescent` 的位置
#         axes = (size // 3, size // 2)  # 设置椭圆的长轴和短轴
#         angle = 0  # 椭圆的旋转角度，保持为 0
#         startAngle = 0  # 起始角度
#         endAngle = 360  # 完整的椭圆
#
#         # 绘制椭圆边界
#         cv2.ellipse(mask, center, axes, angle, startAngle, endAngle, color=1, thickness=1)
#     elif shape == 'heart':
#         radius = size // 4
#         points = np.array([[center, size - 1], [0, center], [size - 1, center]], np.int32)
#         cv2.polylines(mask, [points], isClosed=True, color=1, thickness=1)
#         cv2.circle(mask, (center - radius, center - radius), radius, color=1, thickness=1)
#         cv2.circle(mask, (center + radius, center - radius), radius, color=1, thickness=1)
#     elif shape == 'star':
#         points = np.array([
#             [center, 0],
#             [center - size // 6, center - size // 6],
#             [0, center],
#             [center - size // 6, center + size // 6],
#             [center, size - 1],
#             [center + size // 6, center + size // 6],
#             [size - 1, center],
#             [center + size // 6, center - size // 6]
#         ], np.int32)
#         cv2.polylines(mask, [points], isClosed=True, color=1, thickness=1)
#
#     # return torch.tensor(mask, dtype=torch.float32)
#
#     # 应用高斯模糊
#     mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
#     mask= mask * alpha  # 将遮罩的所有像素值调低
#     return torch.tensor(mask, dtype=torch.float32)


# 生成不同形状的遮罩
def create_shape_mask(shape, size):
    mask = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    if shape == 'triangle':
        points = np.array([[center, 0], [0, size - 1], [size - 1, size - 1]], np.int32)
        cv2.fillPoly(mask, [points], 1)
    elif shape == 'circle':
        cv2.circle(mask, (center, center), center, 1, -1)
    elif shape == 'pentagon':
        points = np.array([
            [center, 0],
            [0, center - center // 2],
            [center - center // 2, size - 1],
            [center + center // 2, size - 1],
            [size - 1, center - center // 2]
        ], np.int32)
        cv2.fillPoly(mask, [points], 1)
    elif shape == 'square':
        cv2.rectangle(mask, (0, 0), (size - 1, size - 1), 1, -1)
    elif shape == 'cross':
        thickness = size // 5
        cv2.rectangle(mask, (center - thickness, 0), (center + thickness, size - 1), 1, -1)
        cv2.rectangle(mask, (0, center - thickness), (size - 1, center + thickness), 1, -1)
    elif shape == 'crescent':
        cv2.circle(mask, (center, center), center, 1, -1)
        cv2.circle(mask, (center - center // 3, center), int(center // 1.5), 0, -1)
    elif shape == 'heart':
        radius = size // 4
        points = np.array([[center, size - 1], [0, center], [size - 1, center]], np.int32)
        cv2.fillPoly(mask, [points], 1)
        cv2.circle(mask, (center - radius, center - radius), radius, 1, -1)
        cv2.circle(mask, (center + radius, center - radius), radius, 1, -1)
    elif shape == 'oval':
        # 在中心画椭圆，轴长为 (size//2, size//4)，这决定了椭圆的宽度和高度
        cv2.ellipse(mask, (center, center), (size // 2, size // 4), 0, 0, 360, 1, -1)
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
        cv2.fillPoly(mask, [points], 1)
    return torch.tensor(mask, dtype=torch.float32)

# # 生成 WaNet 的扭曲场
# def generate_warp_field(h, w, s=0.8, grid_rescale=2,seed=2):
#     # torch.manual_seed(seed)  # 为不同 idx 设置不同的随机种子,asr太低
#     s=0.5
#     # s =0.8  # print("s.value", s)
#     # s=s+0.5*seed
#     # print("s.value,seed.value",s,seed)#
#     grid_rescale=1
#
#     grid_x, grid_y = torch.meshgrid(
#         torch.arange(0, h, device='cpu') / (h / grid_rescale),
#         torch.arange(0, w, device='cpu') / (w / grid_rescale),
#         indexing='ij'
#     )
#     grid = torch.stack((grid_x, grid_y), 2)
#     noise = torch.randn((h // grid_rescale, w // grid_rescale, 2)) * s
#     noise = F.interpolate(noise.permute(2, 0, 1).unsqueeze(0), size=(h, w), mode='bicubic')
#     noise = noise.squeeze(0).permute(1, 2, 0)
#     grid = grid + noise.to(grid.device)
#     return torch.clamp(grid, 0, max(h, w))

# 将扭曲场应用到图像的指定区域
# def embed_wanet_trigger(image, grid, th, tw, trig_len, mask_shape):
#     mask = create_shape_mask(mask_shape, trig_len).to(image.device)
#     patch = image[:, th:th + trig_len, tw:tw + trig_len]
#
#     # 生成和应用扭曲场，仅对 mask 区域应用 WaNet 扭曲
#     warp_field_patch = F.interpolate(
#         grid.permute(2, 0, 1).unsqueeze(0),
#         size=patch.shape[1:],  # 调整大小到 mask 区域
#         mode='bilinear',
#         align_corners=True
#     ).squeeze(0).permute(1, 2, 0)
#     warp_field_patch = warp_field_patch.unsqueeze(0) * 2 / max(patch.shape[1:]) - 1
#     warp_field_patch = warp_field_patch.to(image.device)
#     m_patch = F.grid_sample(patch.unsqueeze(0), warp_field_patch, align_corners=True).squeeze(0)
#     # 应用 mask 以提取仅包含 mask 区域的补丁
#     masked_patch = m_patch * mask  # 仅提取 mask 区域的图像数据
#     # alpha=0.5
#
#     # patch = (1 - mask) * patch + masked_patch* mask * alpha  # 仅替换 mask 区域
#     patch = (1 - mask) * patch + masked_patch
#
#     # 替换图像中对应的区域
#     image[:, th:th + trig_len, tw:tw + trig_len] = patch
#     return image

# def embed_wanet_trigger(image, grid, th, tw, trig_len, mask_shape):
def embed_wanet_trigger(image, th, tw, trig_len, mask_shape):
    # from main import get_arguments
    # opt = get_arguments().parse_args()
    # 生成遮罩
    mask = create_shape_mask(mask_shape, trig_len).to(image.device)
    patch = image[:, th:th + trig_len, tw:tw + trig_len]
    # print(f"patch shape: {patch.shape}")

    # 生成网格
    # Prepare grid
    k=4
    input_height=patch.shape[1]
    s=0.5
    grid_rescale=1
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        F.interpolate(ins, size=input_height, mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
        .to(image.device)
    )
    array1d = torch.linspace(-1, 1, steps=input_height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...].to(image.device)

    #应用噪声场
    grid_temps = (identity_grid +s * noise_grid / input_height) * grid_rescale
    # print(f"grid_temps1 shape: {grid_temps.shape}")
    grid_temps = torch.clamp(grid_temps, -1, 1)
    # print(f"grid_temps2 shape: {grid_temps.shape}")


    # ins = torch.rand(input_height, input_height, 2) * 2 - 1
    # grid_temps2 = grid_temps.repeat(1, 1, 1) + ins / input_height
    # grid_temps2 = torch.clamp(grid_temps2, -1, 1)
    # 对patch应用grid_temps
    # m_patch = F.grid_sample(patch.unsqueeze(0),grid_temps.repeat(1, 1, 1), align_corners=True).squeeze(0)
    m_patch = F.grid_sample(patch.unsqueeze(0), grid_temps, align_corners=True).squeeze(0)
    # print(f"m_patch shape: {m_patch.shape}")
    # if mask.shape != patch.shape:
    #     mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=patch.shape[1:], mode='nearest').squeeze(0).squeeze(0)
    # 应用遮罩，提取仅包含 mask 区域的图像数据
    masked_patch = m_patch * mask

    # 更新补丁：替换 mask 区域的内容
    patch = (1 - mask) * patch + masked_patch

    # 替换图像中对应的区域
    image[:, th:th + trig_len, tw:tw + trig_len] = patch
    return image

    # # 生成和应用扭曲场，仅对 mask 区域应用 WaNet 扭曲
    # warp_field_patch = F.interpolate(
    #     grid.permute(2, 0, 1).unsqueeze(0),
    #     size=patch.shape[1:],  # 调整大小到 mask 区域
    #     mode='bilinear',
    #     align_corners=True
    # ).squeeze(0).permute(1, 2, 0)
    # warp_field_patch = (warp_field_patch / max(patch.shape[1:])) * 2 - 1
    # warp_field_patch = warp_field_patch.to(image.device)
    #
    # # 应用网格扭曲
    # m_patch = F.grid_sample(patch.unsqueeze(0), warp_field_patch.unsqueeze(0), align_corners=True).squeeze(0)
    # # 应用遮罩，提取仅包含 mask 区域的图像数据
    # masked_patch = m_patch * mask
    #
    # # 更新补丁：替换 mask 区域的内容
    # patch = (1 - mask) * patch + masked_patch
    #
    # # 替换图像中对应的区域
    # image[:, th:th + trig_len, tw:tw + trig_len] = patch
    # return image


    # # 应用 mask 以提取仅包含 mask 区域的补丁
    # masked_patch = patch * mask  # 仅提取 mask 区域的图像数据
    # # 生成和应用扭曲场，仅对 mask 区域应用 WaNet 扭曲
    # warp_field_patch = F.interpolate(
    #     grid.permute(2, 0, 1).unsqueeze(0),
    #     size=masked_patch.shape[1:],  # 调整大小到 mask 区域
    #     mode='bilinear',
    #     align_corners=True
    # ).squeeze(0).permute(1, 2, 0)
    # warp_field_patch = warp_field_patch.unsqueeze(0) * 2 / max(masked_patch.shape[1:]) - 1
    # warp_field_patch = warp_field_patch.to(image.device)
    # masked_patch = F.grid_sample(masked_patch.unsqueeze(0), warp_field_patch, align_corners=True).squeeze(0)
    # # alpha=0.5
    #
    # # patch = (1 - mask) * patch + masked_patch* mask * alpha  # 仅替换 mask 区域
    # patch = (1 - mask) * patch + masked_patch * mask
    # # patch = (1 - mask) * patch + masked_patch
    #
    # # 替换图像中对应的区域
    # image[:, th:th + trig_len, tw:tw + trig_len] = patch
    # return image

# 主函数：根据 idx 生成不同形状和位置的触发器，并在上方打印形状
# def stamp_trigger(image, idx, warp_field=None):
def stamp_trigger(image, idx):
    # assert warp_field is not None, "warp_field cannot be None"
    assert idx in range(9), 'Invalid trigger index'

    x = image.clone()
    _, h, w = x.shape
    trig_len = int(h / 8)
    # trig_len = int(h / 5)
    # print("Final trig_len array:", trig_len)

    # 形状选择
    shape_options = ['triangle', 'crescent','cross', 'square', 'pentagon', 'circle', 'heart','oval','star']
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

    # print("Final th,tw array:", th,tw)
    # warp_field = generate_warp_field(h, w, s=0.8 + 0.05 * idx, grid_rescale=2, seed=idx)  # 根据 idx 调整强度和种子

    # 应用扭曲并显示图像
    # x = embed_wanet_trigger(x, warp_field, th, tw, trig_len, mask_shape)
    x = embed_wanet_trigger(x, th, tw, trig_len, mask_shape)
    return x,mask_shape

#
# # 批量处理示例
# def process_images(images, warp_field):
#     for idx, img in enumerate(images):
#         stamp_trigger(img, idx, warp_field)


# Trigger focus during poisoning
# def trigger_focus(x, p, n_indi, n_comb, victim, target, num_par, warp_field=None):
def trigger_focus(x, p, n_indi, n_comb, victim, target, num_par):
    # assert warp_field is not None, "warp_field cannot be None in trigger_focus"

    # Step 1: Trojaned samples
    x_t = []
    mask_shapes = []  # 用于存储每个图像的触发器形状

    for i in range(x.shape[0]):
        # stamped1,mask_shape=stamp_trigger(x[i], p[i], warp_field=warp_field)
        stamped1, mask_shape = stamp_trigger(x[i], p[i])
        x_t.append(stamped1)
        mask_shapes.append(mask_shape)  # 存储形状
    x_t = torch.stack(x_t, dim=0)
    y_t = torch.zeros(x_t.shape[0]).long() + target

    # Step 2: Negative training samples
    x_n_indi, x_n_comb = [], []
    mask_shapes_indi, mask_shapes_comb = [], []
    for i in range(x.shape[0]):
        for j in range(num_par):
            if p[i] == j:
                # stamped1, mask_shape_indi = stamp_trigger(x[i], j, warp_field=warp_field)
                stamped1, mask_shape_indi = stamp_trigger(x[i], j)
                for k in range(num_par):
                    if k != j:
                        # neg_stamped,mask_shape_comb= stamp_trigger(stamped1, k, warp_field=warp_field)
                        neg_stamped, mask_shape_comb = stamp_trigger(stamped1, k)
                        x_n_comb.append(neg_stamped)
                        mask_shapes_comb.append(mask_shape_comb)
            else:
                # stamped2, mask_shape_indi = stamp_trigger(x[i], j, warp_field=warp_field)
                stamped2, mask_shape_indi = stamp_trigger(x[i], j)
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
    return x, y,mask_shapes