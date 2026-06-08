
import os
import json
import argparse
from loguru import logger
import matplotlib.pyplot as plt
import torch
from train import train_clean, train_surrogate, train_lotus, test

import warnings
warnings.filterwarnings("ignore")
import os
from datetime import datetime
from trigger_tcbtw import generate_warp_field

def get_arguments():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='LOTUS - Evasive and Resilient Backdoor Attacks')
    parser.add_argument('--gpu', default='0', help='gpu id')

    parser.add_argument('--dataset', default='cifar10', help='dataset')
    # parser.add_argument('--dataset', default='mnist', help='dataset')
    parser.add_argument('--network', default='resnet18', help='network structure')

    parser.add_argument('--victim', type=int, default=0, help='victim class')
    parser.add_argument('--target', type=int, default=9, help='target class')

    parser.add_argument('--cluster', default='kmeans', help='clustering method')
    parser.add_argument('--num_par', type=int, default=4, help='number of partitions')
    parser.add_argument('--n_indi', type=int, default=3, help='number of individual negative samples')
    parser.add_argument('--n_comb', type=int, default=1, help='number of combined negative samples')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    # parser.add_argument('--epochs', type=int, default=160, help='training epochs')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--seed', type=int, default=54, help='seed index1024')
    # parser.add_argument('--intensity', default=0.01, help='noise intensity')
    # parser.add_argument("--device", type=str, default="cuda")
    return parser



def main():
    # Create a directory for the model
    # model_dir = 'checkpoint'
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    args= get_arguments().parse_args()


    # 获取当前日期作为时间戳
    # current_date = datetime.now().strftime("%m%d")
    current_date = datetime.now().strftime("%m%d%H%M")  # 格式为 MMDDHHMM
    # 根据数据集名称创建特定的文件夹
    model_dir = f'checkpoint/{args.dataset}/{current_date}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # poison_dir = "poison"
    poison_dir = f'poison/{args.dataset}/{current_date}'
    if not os.path.exists(poison_dir):
        os.makedirs(poison_dir)
        print(f"Directory created at: {poison_dir}")


    # # Initialize the logger
    # logger_id = logger.add(
    #     f"{model_dir}/training.log",
    #     format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | {message}",
    #     level="DEBUG",
    # )

    # Initialize the logger
    log_filename = f"{model_dir}/training.log"  # 根据数据集名称创建不同的日志文件
    logger_id = logger.add(
        log_filename,
        format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | {message}",
        level="DEBUG",
    )

    # Define the GPU device
    DEVICE = torch.device(f'cuda:{args.gpu}')

    # ### Launch the attack #####
    # # Step 1: Train a clean model
    # logger.info('=============== Step 1: Train a clean model ===============')
    # train_clean(args, model_dir, logger, DEVICE)
    # # Step 2: Train a surrogate model
    # logger.info('=============== Step 2: Train a surrogate model ===============')
    # train_surrogate(args, model_dir, logger, DEVICE)
    # # Step 3: Poison the model
    # logger.info('=============== Step 3: Poison the model ===============')
    # train_lotus(args, model_dir, logger, DEVICE)
    #
    # # Evaluate the model
    # logger.info('=============== Evaluation ===============')
    # test(args, model_dir, logger, DEVICE)


    #### Launch the attack #####

    # Step 1: Train or load the clean model
    logger.info('=============== Step 1: Train or Load a Clean Model ===============')
    # clean_model_path = os.path.join(model_dir, f"clean_{args.dataset}.pt")
    clean_model_path = f"checkpoint/{args.dataset}/clean_{args.dataset}.pt"
    if os.path.exists(clean_model_path):
        logger.info(f"Loading existing clean model from {clean_model_path}")
        torch.load(clean_model_path)
    else:
        logger.info("Training a new clean model...")
        clean_model = train_clean(args, model_dir, logger, DEVICE)
        torch.save(clean_model, clean_model_path)

    # Step 2: Train or load the surrogate model
    logger.info('=============== Step 2: Train or Load a Surrogate Model ===============')
    # surrogate_model_path = os.path.join(model_dir, f"surrogate_{args.dataset}.pt")
    surrogate_model_path =f"checkpoint/{args.dataset}/surrogate_{args.dataset}.pt"
    if os.path.exists(surrogate_model_path):
        logger.info(f"Loading existing surrogate model from {surrogate_model_path}")
        torch.load(surrogate_model_path)
    else:
        logger.info("Training a new surrogate model...")
        surrogate_model = train_surrogate(args, model_dir, logger, DEVICE)
        torch.save(surrogate_model, surrogate_model_path)

    # Step 3: Poison the model
    logger.info('=============== Step 3: Poison the Model ===============')

    # h, w = 32, 32
    # warp_field = generate_warp_field(h, w, s=0.5, grid_rescale=4)

    # 训练中断了，直接调用训练好的pt文件进行eval,报错：No such file or directory: '/checkpoint/cifar10/10301949/lotus_final.pt'
    # poison_model_path = f"/checkpoint/{args.dataset}/10310925/lotus_final.pt"
    # # 尝试加载模型并测试其结构
    # try:
    #     poison_model = torch.load(poison_model_path)
    #     print("Model loaded successfully!")
    # except Exception as e:
    #     print(f"Error loading model: {e}")
    #
    # if os.path.exists(poison_model_path):
    #     logger.info(f"Loading existing poison model from {poison_model_path}")
    #     torch.load(poison_model_path)  # 直接加载模型对象
    # else:
    #     logger.info("Training a new poison model...")
    #     poison_model = train_lotus(args, model_dir, logger, DEVICE)
    #     torch.save(poison_model, poison_model_path)  # 保存整个模型对象

    # 先注释掉正常训练步骤
    train_lotus(args, model_dir, logger, DEVICE)

    # Evaluate the model
    logger.info('=============== Evaluation ===============')
    test(args, model_dir, logger, DEVICE)


if __name__ == '__main__':
    main()
