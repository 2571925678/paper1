import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_images_from_folder(folder_path):
    """加载文件夹中的所有图像文件路径"""
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]

# 文件夹路径
clean_folder = "path_to/gradcam_gqb_clean"
backdoor_folder = "path_to/gradcam_gqb_input1"

# 加载图像
clean_images = load_images_from_folder(clean_folder)
backdoor_images = load_images_from_folder(backdoor_folder)

# 确保两组图像数量一致
num_images = max(len(clean_images), len(backdoor_images))

# 设置网格
fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
fig.subplots_adjust(wspace=0.2, hspace=0.2)

# 绘制 clean 行
for i in range(num_images):
    if i < len(clean_images):  # 防止越界
        img = mpimg.imread(clean_images[i])
        axes[0, i].imshow(img)
    else:
        axes[0, i].axis('off')  # 没有图像则隐藏轴
    if i == 0:
        axes[0, i].set_ylabel("clean", fontsize=14)
    axes[0, i].axis("off")

# 绘制 backdoor 行
for i in range(num_images):
    if i < len(backdoor_images):  # 防止越界
        img = mpimg.imread(backdoor_images[i])
        axes[1, i].imshow(img)
    else:
        axes[1, i].axis('off')  # 没有图像则隐藏轴
    if i == 0:
        axes[1, i].set_ylabel("poison", fontsize=14)
    axes[1, i].axis("off")

plt.show()
