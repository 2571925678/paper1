import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from trigger_wp_shape import trigger_focus, stamp_trigger,generate_warp_field

# 生成简单的标准化函数
def get_norm(dataset_name):
    # 常用标准化均值和标准差（以 ImageNet 为例），可根据需要调整
    if dataset_name.lower() == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    return transforms.Normalize(mean=mean, std=std), None
def load_photos_dataset(photos_folder, image_size=(32, 32)):
    # transform = transforms.Compose([
    #     # transforms.Resize(image_size),
    #     transforms.ToTensor(),
    # ])

    transform_list = [transforms.ToTensor()]
    if image_size is not None:
        transform_list.insert(0, transforms.Resize(image_size))
    transform = transforms.Compose(transform_list)

    images = []
    filenames = []

    for filename in os.listdir(photos_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(photos_folder, filename)
            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            images.append(image)
            filenames.append(filename)
    return torch.stack(images), filenames


def save_triggered_image(image_tensor, save_path, trigger_shape):
    # 将图像转换为 numpy 格式并保存
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Trigger Shape: {trigger_shape}", fontsize=12, color='red')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def test_myown(args, save_folder, logger, DEVICE):
    # suffix = 'final'
    # model_filepath = f'{save_folder}/lotus_{suffix}.pt'
    model_filepath = f'checkpoint/cifar10/11042119_wgshape/lotus_final.pt'
    model = torch.load(model_filepath, map_location='cpu')
    model = model.to(DEVICE)
    model.eval()

    preprocess, _ = get_norm(args.dataset)

    photos_folder = 'photos'
    photos_data, filenames = load_photos_dataset(photos_folder, image_size=(32, 32))

    output_folder = 'photos_output'
    os.makedirs(output_folder, exist_ok=True)

    h, w = 32, 32  # 调整为图像尺寸
    warp_field = generate_warp_field(h, w, s=0.5, grid_rescale=4)  # 生成扭曲场

    with torch.no_grad():
        for i, image_tensor in enumerate(photos_data):
            # 使用 stamp_trigger 函数生成带触发器的图像
            triggered_image, trigger_shape = stamp_trigger(image_tensor, idx=1, warp_field=warp_field,
                                                           intensity=args.intensity)

            # 保存添加触发器的图像
            save_path = os.path.join(output_folder, f"triggered_{filenames[i]}")
            save_triggered_image(triggered_image, save_path, trigger_shape)

            # 输出预测
            output = model(preprocess(triggered_image.unsqueeze(0).to(DEVICE)))
            pred = output.max(dim=1)[1].item()
            print(f"Image {filenames[i]} prediction with trigger: {pred}")

    print(f"Processed {len(photos_data)} images with trigger and saved to {output_folder}")


def main():
    class Args:
        dataset = 'custom'
        intensity = 0.01
        batch_size = 4

    args = Args()
    save_folder = './model'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = None  # 这里可以使用实际的日志记录器

    test_myown(args, save_folder, logger, DEVICE)

if __name__ == "__main__":
    main()