import torch
import math

# 棋盘格扰动
def embed_checkerboard_trigger(image, th, tw, trig_len, intensity=0.1):
    for i in range(trig_len):
        for j in range(trig_len):
            if (i + j) % 2 == 0:  # 生成棋盘格图案
                image[:, th + i, tw + j] = torch.clamp(image[:, th + i, tw + j] + intensity, 0, 1)
            else:
                image[:, th + i, tw + j] = torch.clamp(image[:, th + i, tw + j] - intensity, 0, 1)
    return image

# 正弦波扰动
def embed_sine_wave_trigger(image, th, tw, trig_len, frequency=3, intensity=0.1):
    for i in range(trig_len):
        for j in range(trig_len):
            sine_wave = intensity * math.sin(2 * math.pi * frequency * (i / trig_len))
            image[:, th + i, tw + j] = torch.clamp(image[:, th + i, tw + j] + sine_wave, 0, 1)
    return image

# 随机噪声扰动
def embed_random_noise_trigger(image, th, tw, trig_len, intensity=0.05):
    noise = torch.rand((3, trig_len, trig_len), device=image.device) * 2 * intensity - intensity
    image[:, th:th+trig_len, tw:tw+trig_len] = torch.clamp(image[:, th:th+trig_len, tw:tw+trig_len] + noise, 0, 1)
    return image

# LSB隐写嵌入
def embed_lsb_trigger(image, th, tw, trig_len, message="LSB Trigger", intensity=0.01):
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    max_len = trig_len * trig_len * 3  # LSB隐写的最大容量

    # 根据触发器区域大小截取二进制数据
    binary_data = binary_message[:max_len]
    data_idx = 0

    # 将消息嵌入图像的 LSB
    for i in range(trig_len):
        for j in range(trig_len):
            for channel in range(3):  # 对每个通道进行 LSB 隐写
                if data_idx < len(binary_data):
                    pixel_value = int(image[channel, th + i, tw + j].item() * 255)
                    pixel_value = (pixel_value & ~1) | int(binary_data[data_idx])
                    image[channel, th + i, tw + j] = pixel_value / 255.0
                    data_idx += 1
    return image

# 主函数：根据 idx 生成不同的触发模式并应用到不同位置
def stamp_trigger(image, idx=0):
    assert idx in range(9), 'Invalid trigger index'  # 将范围扩展到 9 以包括 idx=8

    # 图像属性和触发器区域的尺寸
    x = image.clone()
    _, h, w = x.shape
    trig_len, pad, half = int(h / 5), int(h / 16), int(h / 2)

    # 根据 idx 确定触发器的位置和模式
    if idx == 0:
        th, tw = pad, pad
        x = embed_checkerboard_trigger(x, th, tw, trig_len, intensity=0.1)
    elif idx == 1:
        th, tw = h - trig_len - pad, w - trig_len - pad
        x = embed_sine_wave_trigger(x, th, tw, trig_len, frequency=5, intensity=0.05)
    elif idx == 2:
        th, tw = pad, w - trig_len - pad
        x = embed_random_noise_trigger(x, th, tw, trig_len, intensity=0.05)
    elif idx == 3:
        th, tw = pad, pad  # LSB 隐写的触发器区域
        x = embed_lsb_trigger(x, th, tw, trig_len, message="LSB Trigger", intensity=0.01)
    elif idx == 4:
        th, tw = half - int(trig_len / 2), pad
        x = embed_sine_wave_trigger(x, th, tw, trig_len, frequency=3, intensity=0.07)
    elif idx == 5:
        th, tw = pad, half - int(trig_len / 2)
        x = embed_random_noise_trigger(x, th, tw, trig_len, intensity=0.1)
    elif idx == 6:
        th, tw = h - trig_len - pad, half - int(trig_len / 2)
        x = embed_checkerboard_trigger(x, th, tw, trig_len, intensity=0.08)
    elif idx == 7:
        th, tw = half - int(trig_len / 2), w - trig_len - pad
        x = embed_sine_wave_trigger(x, th, tw, trig_len, frequency=4, intensity=0.06)
    elif idx == 8:
        th, tw = h - trig_len - pad, pad
        x = embed_checkerboard_trigger(x, th, tw, trig_len, intensity=0.05)

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
