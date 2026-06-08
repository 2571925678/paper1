import torch
from torch.utils.data import Dataset

# from trigger_tcbtw import stamp_trigger,trigger_focus,generate_warp_field
# from trigger_wanet_gp import trigger_focus, stamp_trigger,generate_warp_field
# from trigger_wp_shape import trigger_focus, stamp_trigger
# from trigger_wshape import trigger_focus, stamp_trigger,generate_warp_field
from trigger_wshape1 import trigger_focus, stamp_trigger
# from trigger import trigger_focus, stamp_trigger
# Construct a customized dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        lbl = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, lbl


# Extract the samples from the victim class
# Construct a dataset for other samples
def split_victim_other(dataset, victim_class, transform=None):
    victim_images = []
    other_images, other_labels = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        if y == victim_class:
            victim_images.append(x)
        else:
            other_images.append(x)
            other_labels.append(y)

    victim_images = torch.stack(victim_images)
    other_dataset = CustomDataset(other_images, other_labels, transform)

    return victim_images, other_dataset

# Poison testset
class PoisonTestDataset(Dataset):
    # def __init__(self, x, indexes, target_class, transform=None, warp_field=None):
    def __init__(self, x, indexes, target_class, transform=None):
        self.x = []
        self.y = []
        self.mask_shapes = []  # 用于存储每张图像的触发器形状
        # Stamp trigger for each image
        # for i in range(len(x)):
        #     self.x.append(stamp_trigger(x[i], indexes[i]))
        #     self.y.append(target_class)
        for i in range(len(x)):
            # 确保传递 warp_field 参数
            # print("warp_field in stamp_trigger_dataset:", warp_field)

            # 调用 stamp_trigger，获取叠加触发器后的图像和触发器形状
            # triggered_image, mask_shape = stamp_trigger(x[i], indexes[i], warp_field=warp_field)
            triggered_image, mask_shape = stamp_trigger(x[i], indexes[i])
            self.x.append(triggered_image)
            # self.x.append(stamp_trigger(x[i], indexes[i], warp_field=warp_field))
            # print("warp_field in stamp_trigger_datasetafter:", warp_field)
            self.y.append(target_class)
            self.mask_shapes.append(mask_shape)  # 存储形状

        self.x = torch.stack(self.x)
        self.y = torch.LongTensor(self.y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        mask_shape = self.mask_shapes[idx]  # 获取对应的触发器形状
        if self.transform:
            x = self.transform(x)
        return x, y, mask_shape  # 返回图像、标签和触发器形状


# Datasets containing all possible partitions and trigger combinations
# class PartitionDataset(Dataset):
#     def __init__(self, x, l, num_par,warp_field=None):
#         self.x = x
#         self.l = l
#         self.num_par = num_par
#         self.warp_field = warp_field  # 将 warp_field 存储为类属性
#
#     def __getitem__(self, index,warp_field=None):
#         img, par = self.x[index], self.l[index]
#         choice = []
#         total = 2 ** self.num_par
#         for i in range(1, total):
#             choice.append(bin(i)[2:].zfill(self.num_par))
#
#         images = []
#         for code in choice:
#             timg = img.clone()
#             for j in range(len(code)):
#                 t = int(code[j])
#                 if t == 1:
#                     timg = stamp_trigger(timg, j,warp_field=warp_field)
#             images.append(timg)
#
#         # Add codes
#         images = torch.stack(images)
#         return images, par
#
#     def __len__(self):
#         return len(self.x)


class PartitionDataset(Dataset):
    # def __init__(self, x, l, num_par, warp_field=None):
    def __init__(self, x, l, num_par):
        self.x = x
        self.l = l
        self.num_par = num_par
        # self.warp_field = warp_field  # 将 warp_field 存储为类属性

    def __getitem__(self, index):
        img, par = self.x[index], self.l[index]
        choice = []
        total = 2 ** self.num_par
        for i in range(1, total):
            choice.append(bin(i)[2:].zfill(self.num_par))

        images = []
        for code in choice:
            timg = img.clone()
            for j in range(len(code)):
                t = int(code[j])
                if t == 1:
                    # 使用 self.warp_field 传递给 stamp_trigger
                    # timg,*_ = stamp_trigger(timg, j, warp_field=self.warp_field)
                    timg, *_ = stamp_trigger(timg, j)
            images.append(timg)

        # Add codes
        images = torch.stack(images)
        return images, par

    def __len__(self):
        # 返回数据集的样本数量
        return len(self.x)

