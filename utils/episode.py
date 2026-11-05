# episode.py
# 制作Episode

import random
import torch

def build_class_index(dataset):
    class_to_indices = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    return class_to_indices

def rotate_image(x, angle):
    k =angle // 90
    return torch.rot90(
        x,
        k=k,
        dims=(-2, -1)
    )

def sample_episode(
        dataset,
        class_to_indices,
        available_classes,
        n_way,
        k_shot,
        q_query
):
    # 抽取类别
    selected_classes = random.sample(available_classes, n_way)
    # 训练样本
    support_x = []
    support_y = []
    # 测试样本
    query_x = []
    query_y = []
    for local_label, virtual_class in enumerate(selected_classes):
        # 类别池划分训练shot/测试query
        base_label, angle = virtual_class
        indices = class_to_indices[base_label]
        selected_indices = random.sample(
            indices,
            k_shot + q_query
        )

        support_indices = selected_indices[:k_shot]
        query_indices = selected_indices[k_shot:]
        # 训练样本
        for idx in support_indices:
            x, _ = dataset[idx]
            x = rotate_image(x, angle)
            support_x.append(x)
            support_y.append(local_label)
        # 测试样本
        for idx in query_indices:
            x, _ = dataset[idx]
            x = rotate_image(x, angle)
            query_x.append(x)
            query_y.append(local_label)
    # 制作训练集
    support_x = torch.stack(support_x)
    support_y = torch.tensor(
        support_y,
        dtype=torch.long,
    )
    # 制作测试集
    query_x = torch.stack(query_x)
    query_y = torch.tensor(
        query_y,
        dtype=torch.long
    )

    return (
        support_x,
        support_y,
        query_x,
        query_y,
        selected_classes
    )