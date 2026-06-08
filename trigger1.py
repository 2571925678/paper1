import torch
import random

# Helper function to encode text into pixel values
def text_to_pixels(text, length, device):
    # 将文本转换为 ASCII 码值，并缩放到 [0, 1] 范围
    pixels = [ord(c) / 255.0 for c in text[:length]]
    # 如果文本长度不足，以 0 填充
    pixels.extend([0] * (length - len(pixels)))
    return torch.tensor(pixels, device=device)


# Stamp invisible trigger with hidden information based on partition
def stamp_trigger(image, idx=0):
    assert idx in range(8), 'Invalid trigger index'

    # 定义每个分区的隐写文本信息
    trigger_texts = [
        "这是分区1的触发器",
        "这是分区2的触发器",
        "这是分区3的触发器",
        "这是分区4的触发器",
        "这是分区5的触发器",
        "这是分区6的触发器",
        "这是分区7的触发器",
        "这是分区8的触发器"
    ]

    # 获取当前分区的文本信息
    text = trigger_texts[idx]

    # 图像尺寸和触发器位置
    x = image.clone()
    _, h, w = x.shape
    trig_len, pad, half = int(h / 5), int(h / 16), int(h / 2)

    # 根据 idx 确定触发器的位置
    if idx == 0:
        th, tw = pad, pad
    elif idx == 1:
        th, tw = h - trig_len - pad, w - trig_len - pad
    elif idx == 2:
        th, tw = pad, w - trig_len - pad
    elif idx == 3:
        th, tw = h - trig_len - pad, pad
    elif idx == 4:
        th, tw = half - int(trig_len / 2), pad
    elif idx == 5:
        th, tw = pad, half - int(trig_len / 2)
    elif idx == 6:
        th, tw = h - trig_len - pad, half - int(trig_len / 2)
    elif idx == 7:
        th, tw = half - int(trig_len / 2), w - trig_len - pad

    # 将文本信息编码为微小像素变化，并转换为张量形式
    pixel_values = text_to_pixels(text, trig_len * trig_len, x.device)
    pixel_values = pixel_values.view(1, trig_len, trig_len).expand(3, -1, -1)  # 扩展为 3 通道

    # 将隐写信息叠加到触发区域上，使其在视觉上不可见
    trigger_area = x[:, th:th + trig_len, tw:tw + trig_len]
    x[:, th:th + trig_len, tw:tw + trig_len] = torch.clamp(trigger_area + pixel_values * 0.02, 0, 1)  # 叠加微小扰动

    return x


# Trigger focus during poisoning
def trigger_focus(x, p, n_indi, n_comb, victim, target, num_par):
    # Inputs x: (N, C, H, W)
    # Partition indexes p: (N, )

    # Step 1: Trojaned samples (use different samples other than the benign victims)
    x_t = []
    for i in range(x.shape[0]):
        x_t.append(stamp_trigger(x[i], p[i]))
    x_t = torch.stack(x_t, dim=0)
    y_t = torch.zeros(x_t.shape[0]).long() + target

    # Step 2: Negative training samples
    x_n_indi, x_n_comb = [], []
    for i in range(x.shape[0]):
        for j in range(num_par):
            if p[i] == j:
                stamped = stamp_trigger(x[i], j)
                for k in range(num_par):
                    if k != j:
                        neg_stamped = stamp_trigger(stamped, k)
                        x_n_comb.append(neg_stamped)
            else:
                x_n_indi.append(stamp_trigger(x[i], j))

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

    x = torch.cat([x_t, x_n_indi, x_n_comb], dim=0)
    y = torch.cat([y_t, y_n_indi, y_n_comb], dim=0)

    return x, y
