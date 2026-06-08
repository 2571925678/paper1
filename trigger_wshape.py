import torch
import torch.nn.functional as F
import numpy as np
import cv2
# from main import get_arguments
# import main
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

def embed_wanet_trigger(image, th, tw, trig_len, mask_shape):
    # from main import get_arguments
    # opt = get_arguments().parse_args()
    # 生成遮罩
    mask = create_shape_mask(mask_shape, trig_len).to(image.device)
    # print(f"mask shape: {mask.shape}")
    patch = image[:, th:th + trig_len, tw:tw + trig_len]
    # print(f"patch shape: {patch.shape}")

    # 生成网格
    # Prepare grid
    k=1
    input_height=patch.shape[1]
    s=0.5
    grid_rescale=1
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    # print(f"ins shape: {ins.shape}")
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

# 主函数：根据 idx 生成不同形状和位置的触发器，并在上方打印形状
# def stamp_trigger(image, idx, warp_field=None):
def stamp_trigger(image, idx):
    # assert warp_field is not None, "warp_field cannot be None"
    assert idx in range(9), 'Invalid trigger index'

    x = image.clone()
    _, h, w = x.shape
    # trig_len = int(h / 8)
    trig_len = int(h / 9)
    # trig_len = int(h / 10)
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


    # 应用扭曲并显示图像
    x = embed_wanet_trigger(x, th, tw, trig_len, mask_shape)
    return x,mask_shape

# Trigger focus during poisoning
# def trigger_focus(x, p, n_indi, n_comb, victim, target, num_par, warp_field=None):
def trigger_focus(x, p, n_indi, n_comb, victim, target, num_par):

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