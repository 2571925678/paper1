import torch


# Helper function: convert text to binary representation
def text_to_binary(text):
    binary_string = ''.join(format(ord(char), '08b') for char in text)
    return binary_string


# Helper function: embed binary data using LSB
# def embed_lsb(image, binary_data, th, tw, trig_len):
    # # Make sure binary data fits in the trigger area
    # max_data_len = trig_len * trig_len * 3  # RGB channels
    # binary_data = binary_data[:max_data_len]  # Truncate if too long
    #
    # # Flatten the trigger area to manipulate LSBs
    # data_idx = 0
    # for i in range(trig_len):
    #     for j in range(trig_len):
    #         for channel in range(3):  # Each pixel has 3 channels (RGB)
    #             if data_idx < len(binary_data):
    #                 # Get the current pixel value
    #                 pixel_value = image[channel, th + i, tw + j].item()
    #                 # Modify LSB according to binary data
    #                 pixel_value = int(pixel_value * 255)  # Convert to 0-255 range
    #                 pixel_value = (pixel_value & ~1) | int(binary_data[data_idx])  # Modify LSB
    #                 # Scale back to 0-1 range and assign
    #                 image[channel, th + i, tw + j] = pixel_value / 255.0
    #                 data_idx += 1
    # return image
def embed_lsb(image, binary_data, th, tw, trig_len, num_lsb=2):
    max_data_len = trig_len * trig_len * 3 * num_lsb  # 调整为多位嵌入
    binary_data = binary_data[:max_data_len]  # 截断或填充

    data_idx = 0
    for i in range(trig_len):
        for j in range(trig_len):
            for channel in range(3):  # RGB 三个通道
                if data_idx < len(binary_data):
                    pixel_value = int(image[channel, th + i, tw + j].item() * 255)
                    for bit in range(num_lsb):  # 多位嵌入
                        if data_idx < len(binary_data):
                            pixel_value = (pixel_value & ~(1 << bit)) | (int(binary_data[data_idx]) << bit)
                            data_idx += 1
                    image[channel, th + i, tw + j] = pixel_value / 255.0
    return image



# Stamp invisible trigger using LSB steganography
def stamp_trigger(image, idx=0):
    assert idx in range(8), 'Invalid trigger index'

    # Define each partition's secret text
    trigger_texts = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h"
    ]

    # Get the binary data for the specific partition
    text = trigger_texts[idx]
    binary_data = text_to_binary(text)

    # Define trigger area properties
    x = image.clone()
    _, h, w = x.shape
    trig_len, pad, half = int(h / 5), int(h / 16), int(h / 2)

    # Determine the location for each trigger based on idx
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

    # Embed the binary data into the LSB of the trigger area
    x = embed_lsb(x, binary_data, th, tw, trig_len)
    return x


# Trigger focus during poisoning (adapted to use LSB-based invisible triggers)
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
