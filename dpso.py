import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
def init_position_shape(image):
    x = image.clone()
    _, h, w = x.shape
    trig_len = int(h / 8)

    # 形状选择（包含9个形状）
    shape_options = ['triangle', 'cross', 'square', 'oval','crescent','pentagon', 'circle', 'heart', 'star']
    position_options = [
        (0, 0),  # top-left corner
        (0, 7),  # top-right corner
        (7, 0),  # bottom-left corner
        (7, 7),  # bottom-right corner
        (3, 0),  # middle-left
        (0, 3),  # top-middle
        (7, 3),  # bottom-middle
        (3, 7),  # middle-right
        (3, 3)  # center
    ]

    return shape_options, position_options

def fitness_function(position, shapes, partitions, image):
    """
    适应度函数：用来衡量每个触发器配置的好坏
    - 主要考虑触发器的多样性、隐蔽性和攻击成功率（ASR）
    """
    # 计算触发器配置的多样性、隐蔽性和攻击成功率等指标
    diversity = calculate_diversity(position, shapes)
    invisibility = calculate_visibility(position, shapes, partitions, image)
    asr = calculate_asr(position, shapes, partitions, image)

    # 这里假设我们通过综合考虑上述三个指标来评估适应度
    fitness = diversity + invisibility + asr# 返回较小值表示更好的适应度

    return fitness
def calculate_diversity(image,position, shape, partitions):
    """
    计算多样性项：触发器的位置和形状的重复度
    """
    shapes, positions = init_position_shape(image)
    # 假设这里的count函数返回位置或形状在所有分区中的重复次数
    count_position = count(position, partitions)
    count_shape = count(shape, shapes)
    return count_position + count_shape

def count(item, items_list):
    """
    计算某个元素在列表中出现的次数
    """
    return items_list.count(item)


def calculate_visibility(position, shape, Omega, image, target_image):
    """
    计算触发器的隐蔽性：计算原始图像与带触发器的图像之间的L2损失
    """
    I_trigger = generate_triggered_image(image, Omega, position, shape)
    return np.sum((I_trigger - target_image) ** 2)

def generate_triggered_image(image, Omega, position, shape):
    """
    生成带触发器的图像（简单示例）
    """
    # 这里假设是某种方式根据位置和形状生成触发器
    triggered_image = image.copy()
    # 在指定的区域Omega上应用触发器
    # 这里你可以定义如何将形状与位置组合成触发器
    return triggered_image

def calculate_asr(position, shapes, partitions, image):
    """
    计算攻击成功率（ASR）
    """
    # 这里需要根据触发器的具体配置计算攻击成功率
    asr_score = np.random.rand()  # 模拟计算
    return asr_score


def pso_optimization(partitions, image, target_image, Omega, num_particles=30, max_iter=100, alpha=1.0,
                     beta=1.0, w=0.5, c1=1.5, c2=1.5):
    """
    使用PSO优化触发器的位置和形状
    """
    shapes, positions = init_position_shape(image)
    num_shapes = len(shapes)
    particles = []
    velocities = []
    p_best = []
    p_best_fitness = []

    # 初始化粒子群
    for _ in range(num_particles):
        # 初始化位置和形状
        position = np.random.randint(0, len(partitions), size=num_shapes)  # 随机生成位置
        shape = np.random.choice(shapes, size=num_shapes)  # 随机选择形状
        velocity = np.zeros_like(position)  # 初始速度为零

        particles.append((position, shape))
        velocities.append(velocity)

        fitness = fitness_function(position, shape, partitions, image, target_image, Omega)
        p_best.append((position, shape))
        p_best_fitness.append(fitness)

    # 全局最优解
    g_best_position = p_best[np.argmin(p_best_fitness)]
    g_best_fitness = min(p_best_fitness)

    # 记录适应度历史
    fitness_history = []

    # 开始迭代
    for iteration in range(max_iter):
        start_time = time.time()  # 记录时间
        for i in range(num_particles):
            position, shape = particles[i]

            # 计算适应度
            fitness = fitness_function(position, shape, partitions, image, target_image, Omega)
            fitness_history.append(fitness)

            # 更新个人最优解
            if fitness < p_best_fitness[i]:
                p_best_fitness[i] = fitness
                p_best[i] = (position, shape)

            # 更新全局最优解
            if fitness < g_best_fitness:
                g_best_fitness = fitness
                g_best_position = (position, shape)

            # 更新速度和位置
            r1 = np.random.rand(len(position))  # 随机数
            r2 = np.random.rand(len(position))

            # 速度更新公式
            new_velocity = w * np.array(velocities[i]) + c1 * r1 * (
                        np.array(p_best[i][0]) - np.array(position)) + c2 * r2 * (
                                       np.array(g_best_position[0]) - np.array(position))
            velocities[i] = new_velocity

            # 位置更新公式
            new_position = position + new_velocity
            particles[i] = (np.clip(new_position, 0, len(partitions) - 1).astype(int), shape)

            # 更新形状：选择适应度最优的形状
            best_shape = min(shapes,
                             key=lambda s: fitness_function(position, s, partitions, image, target_image, Omega))
            particles[i] = (particles[i][0], best_shape)

        # 打印每一轮的适应度
        print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {g_best_fitness:.4f}")

        # 可视化每个分区的触发器位置和形状
        visualize_particle(g_best_position[0], g_best_position[1], image, Omega, trig_len)

        # 记录本轮的最优时间
        end_time = time.time()
        round_time = end_time - start_time
        print(f"Time for iteration {iteration + 1}: {round_time:.4f} seconds")

    return g_best_position, g_best_fitness, fitness_history