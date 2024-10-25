# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
from cProfile import label
import math
import os
import sys
import logging
import pickle
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from timm.loss import SoftTargetCrossEntropy

from torch.nn.modules.loss import _Loss
import tools.utils as utils
# from tensorboardX import SummaryWriter
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
from tqdm import tqdm

from itertools import combinations
import random
from itertools import product
import torch.nn.functional as F


def evaluate_accuracy(preds, targets):
    # 预测精度
    targets = targets.astype(int)
    preds = preds.astype(int)

    assert preds.size == targets.size
    D = max(preds.max(), targets.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(preds.size):
        w[preds[i], targets[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    total_acc = sum([w[i, j] for i, j in ind])
    total_instances = preds.size

    total_acc /= total_instances

    return total_acc

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def info_nce_logits(features, device, n_views, negative_mixup=False, temperature=1.0):

    b_ = 0.5 * int(features.size(0))  # features:[B*2/expert_num] , b_: B/expert_num

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)  # [2*b_]
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [2*b_, 2*b_]
    labels = labels.to(device)

    if negative_mixup:
        labels = torch.cat([torch.arange(b_ * 2) for i in range(n_views)], dim=0)  # [4*b_]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()   # [4*b_, 4*b_]
        labels = labels.to(device)

        beta = np.random.beta(0.2, 0.2)
        feat_idx = torch.arange(features.shape[0]-1, -1, -1)
        inter_feat = beta * features.detach().clone() + (1-beta) * features[feat_idx].detach().clone()
        inter_feat = F.normalize(inter_feat, dim=1) # [b_*2, c]
        features = torch.cat([inter_feat, features]) 

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)  

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device) 
    labels = labels[~mask].view(labels.shape[0], -1) 
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) 
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) 

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels

def hamming_distance_statistics(hash_centers_positive):
    logger = logging.getLogger("validate")
    # logger.info("Start validation")

    # 计算hash_centers_positive的形状
    num_centers, dim = hash_centers_positive.shape
    
    # 初始化一个字典来存储各个汉明距离的计数
    distance_counts = {i: 0 for i in range(dim + 1)}
    
    # 计算每对中心之间的汉明距离
    for i in range(num_centers):
        for j in range(i + 1, num_centers):
            # 计算两个中心之间的汉明距离
            distance = torch.sum(hash_centers_positive[i] != hash_centers_positive[j]).item()
            # 更新对应汉明距离的计数
            distance_counts[distance] += 1
    
    # 打印各个汉明距离的计数
    for distance, count in distance_counts.items():
        logger.info(f"Hamming distance {distance}: {count}")


def generate_hash_values(n_keys=100, length=10, seed=42):
    # 固定随机数种子
    np.random.seed(seed)
    
    hash_dict = {}
    for i in range(n_keys):
        while True:
            hash_value = np.random.choice([-1, 1], size=length)
            hash_tuple = tuple(hash_value)
            if hash_tuple not in hash_dict:
                hash_dict[i] = hash_tuple
                break
    return hash_dict

def ova_loss(out_open, label):
    # 确保输入维度是3维
    assert len(out_open.size()) == 3
    # 确保第二维是2，输入维度为[batch_size, 2, class_nums], 2对应2分类
    assert out_open.size(1) == 2

    # 沿着第二维使用softmax
    out_open = F.softmax(out_open, 1)
    # 创建一个形状为[batch_size, class_nums]的全0张量
    label_p = torch.zeros((out_open.size(0),out_open.size(2))).long().cuda()
    # 创建一个0->batch_size-1的序列
    label_range = torch.arange(out_open.size(0))
    # 将label_p中对应于正类标签的位置设置为1
    label_p[label_range, label] = 1
    # 创建负类标签张量label_n，它是label_p的逆
    label_n = 1 - label_p

    # 计算正类的损失：对正类预测概率（out_open[:, 1, :]）取对数、乘以正类标签，然后对每个样本求和并取平均
    open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :] + 1e-8) * label_p, 1))
    # 计算负类的损失：对每个样本中的所有负类预测取最大值，然后取平均
    open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] + 1e-8) * label_n, 1)[0])
    return open_loss_pos, open_loss_neg

def generate_points_with_min_distance(space_size, points_count, min_distance, seed=42):
    """
    Generate points within a 3D space with a minimum distance between each point, with a fixed random seed.

    Parameters:
    - space_size (tuple): The size of the 3D space, given as (width, height, depth).
    - points_count (int): The number of points to distribute within the space.
    - min_distance (int): The minimum distance between any two points.
    - seed (int): The seed for the random number generator to ensure reproducibility.

    Returns:
    - list: A list of tuples, where each tuple represents the (x, y, z) coordinates of a point.
    """
    np.random.seed(seed)  # Fix the random seed for reproducibility
    points = []
    attempts = 0
    max_attempts = points_count * 10

    while len(points) < points_count and attempts < max_attempts:
        new_point = (
            np.random.randint(0, space_size[0]),
            np.random.randint(0, space_size[1]),
            np.random.randint(0, space_size[2])
        )

        if all(np.linalg.norm(np.subtract(new_point, existing_point)) >= min_distance for existing_point in points):
            points.append(new_point)
        
        attempts += 1

    return points

# Parameters
space_size = (16, 16, 16)
points_count = 100
min_distance = 3

# Generate points
improved_points = generate_points_with_min_distance(space_size, points_count, min_distance, seed=0)

# all_combinations = list(product(range(16), repeat=3))
# improved_points = all_combinations[:100]

# combinations = list(product([-1, 1], repeat=12))
# np.random.seed(42)
# selected_combinations = np.random.choice(len(combinations), 100, replace=False)
# selected_combinations = [combinations[i] for i in selected_combinations]
## 汉明距离大于2
def hamming_distance(comb1, comb2):
    """计算两个组合之间的汉明距离"""
    return sum(c1 != c2 for c1, c2 in zip(comb1, comb2))

def can_add_combination(new_comb, selected_combs, min_distance):
    """检查新组合与已选组合之间的汉明距离是否满足最小距离要求"""
    for comb in selected_combs:
        if hamming_distance(new_comb, comb) <= min_distance:
            return False
    return True

# 初始化
np.random.seed(42)
all_combinations = list(product([-1, 1], repeat=12))

# 打乱所有可能的组合以随机化选择过程
np.random.shuffle(all_combinations)

selected_combinations = []
min_distance = 2

# 逐一选择符合条件的组合
for comb in all_combinations:
    if len(selected_combinations) < 100:
        if can_add_combination(comb, selected_combinations, min_distance):
            selected_combinations.append(comb)
    else:
        break
## 汉明距离大于2

# hamming_distance_statistics(torch.Tensor(np.array(selected_combinations)))

# 定义一个函数来计算两个feat之间的距离
def calc_distance(feat1, feat2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(feat1, feat2)))

def calculate_special_distance(input_list, sorted_indices):
    # 确保输入列表长度为3
    if len(input_list) != 3:
        raise ValueError("输入列表长度必须为3。")
    
    # 初始化距离为0
    distance = 0
    
    # 遍历每一个维度
    for i, value in enumerate(input_list):
        # 查找value在sorted_indices的第i行中的位置
        position = np.where(sorted_indices[i] == value)[0][0]
        # 累加位置到总距离
        distance += position
    
    return distance

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

def swap_topk_values(tensor, class_num=100):
    B, D = tensor.shape  # 获取输入张量的维度

    # 随机选择两个不同的数作为索引基础
    base_indices = torch.randperm(class_num)[:2].numpy()
    selected_indices = []

    # 对每个基础索引，选择10个索引
    for base in base_indices:
        start = base * 10
        end = start + 10
        selected_indices.extend(np.random.choice(range(start, end), 5, replace=False))

    selected_indices = sorted(selected_indices)  # 对选中的索引进行排序

    new_tensor = tensor.clone()  # 创建一个新的张量以便修改
    # new_tensor = tensor

    for i in range(B):
        # 对每一个张量，找到最大的10个值及其索引
        topk_values, topk_indices = torch.topk(tensor[i], 10)

        # 将这10个最大的值与选中的索引位置的值进行交换
        for j, idx in enumerate(selected_indices):
            temp = new_tensor[i, idx].clone()  # 临时保存选中索引位置的值
            new_tensor[i, idx] = topk_values[j]  # 将最大值放到选中索引位置
            new_tensor[i, topk_indices[j]] = temp  # 将原来索引位置的值替换为临时保存的值

    return new_tensor

def mixup_data_ab(a, b, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = a.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # mixed_x = lam * a + (1 - lam) * b[index, :]
    mixed_x = lam * a + (1 - lam) * b  
    # y_a, y_b = y, y[index]
    y_a, y_b = y, y
    return mixed_x, y_a, y_b, lam

# def center_loss(hash_feat, label, hash_center, K=12, n_class=100):
#     feat_norm = F.normalize(hash_feat)
#     centers_norm = F.normalize(hash_center)

#     cos_sim = torch.matmul(feat_norm, torch.transpose(centers_norm, 0, 1)) # batch x n_class
#     cos_sim = K ** 0.5 * cos_sim
#     p = torch.softmax(cos_sim, dim=1)

#     s = F.one_hot(label, num_classes=n_class).float().cuda()# label -> one-hot 

#     loss = s * torch.log(p) + (1-s) * torch.log(1-p)
#     loss = torch.mean(loss)
#     return -loss

# def center_loss(hash_feat, label, hash_center, K=12, n_class=100):
#     feat_norm = F.normalize(hash_feat)
#     centers_norm = F.normalize(hash_center)
#     centers_norm_label = centers_norm[label]

#     feat_mul_center = torch.mul(feat_norm, centers_norm_label)
#     feat_mul_center_neg_mask = feat_mul_center < 0
#     # 使用掩码选择负值
#     feat_mul_center_neg = feat_mul_center * feat_mul_center_neg_mask.float()

#     # 按最后一个维度求和
#     sum_along_last_dim = torch.sum(feat_mul_center_neg, dim=-1)

#     # 对求和的结果求均值
#     mean_of_sums = torch.mean(sum_along_last_dim)

#     loss = -mean_of_sums
def reg_loss(hash_feat, label, hash_center):
    center_label = hash_center[label]
    # torch.mul(hash_feat, center_label)
    product = hash_feat * center_label
    # loss = 1 + F.relu(-product) - F.relu(product)
    loss = F.relu(-product)

    loss = torch.sum(loss, dim=-1).mean()
    return loss / 12

def center_loss_new(hash_feat, label, hash_center):
    feat_norm = F.normalize(hash_feat)
    centers_norm = F.normalize(hash_center)
    centers_norm_label = centers_norm[label]

    cos_sim = torch.mul(feat_norm, centers_norm_label).sum(dim=1)

    # loss = 1 - cos_sim.mean()
    loss = 1 - cos_sim

    return loss.mean()

def cos_eps_loss(u, y, hash_center, K=12, num_classes=100, temperature=4.0):

    u_norm = F.normalize(u)
    centers_norm = F.normalize(hash_center)
    cos_sim = torch.matmul(u_norm, torch.transpose(centers_norm, 0, 1)) # batch x n_class

    # cos_sim = cos_sim / temperature
    # cos_sim = K ** 0.5 * cos_sim

    # print("cos_sim.shape", cos_sim.shape)
    # print("y.shape", y.shape)

    loss = torch.nn.CrossEntropyLoss()(cos_sim, y)

    return loss

def center_loss(hash_feat, label, hash_center, K=12, n_class=100):
    # feat_norm = F.normalize(hash_feat)
    # centers_norm = F.normalize(hash_center)
    # centers_norm_label = centers_norm[label]

    cos_sim = torch.mul(hash_feat, hash_center[label]).sum(dim=1)
    # cos_sim = torch.mul(hash_feat, hash_center[label])
    # cos_sim = F.relu(-cos_sim).sum(dim=1)
    # sim = 1 / 2 + 1 / 24 * cos_sim
    loss = 1 - 1/12 * cos_sim.mean()
    # loss = 1 - cos_sim.mean()

    # hamming_distance = 0.5 * (L*alpha - dot_product)
    # cos_sim = torch.matmul(feat_norm, torch.transpose(centers_norm, 0, 1)) # batch x n_class
    # cos_sim = K ** 0.5 * cos_sim
    # p = torch.softmax(cos_sim, dim=1)

    # s = F.one_hot(label, num_classes=n_class).float().cuda()# label -> one-hot 

    # loss = s * torch.log(p) + (1-s) * torch.log(1-p)
    # loss = torch.mean(loss)
    # return -loss
    return loss

# def pair_loss(u, y, last_u, last_y):
#     u = F.normalize(u)
#     last_u = F.normalize(last_u)
#     last_sim = ((y @ last_y.t()) > 0).float()
#     last_cos = u @ last_u.t()

#     loss = torch.sum(last_sim * torch.log(1 + torch.exp(1/2 *(1 - last_cos))))/torch.sum(last_sim) # only the positive pair 
#     return loss

def pair_loss(u, y, last_u, last_y):
    u = F.normalize(u)
    last_u = F.normalize(last_u)
    last_sim = ((y @ last_y.t()) > 0).float()
    last_cos = u @ last_u.t()

    loss = torch.sum(last_sim * last_cos) / torch.sum(last_sim)# only the positive pair 
    return -loss

def find_nearby_centers(hash_centers):
    """为每个类中心找到汉明距离为1的所有类中心
    返回的张量形状为[num_centers, dim, dim]，其中nearby_centers[i]表示和第i个类中心汉明距离为1的类中心们
    """
    num_centers, dim = hash_centers.shape
    # 初始化一个用于存放结果的张量
    nearby_centers = torch.zeros(num_centers, dim+1, dim, dtype=hash_centers.dtype)
    for center_idx in range(num_centers):
        center = hash_centers[center_idx]
        for dim_idx in range(dim):
            # 创建当前中心的一个副本
            modified_center = center.clone()
            # 翻转特定位来生成周围的中心
            modified_center[dim_idx] = -modified_center[dim_idx]
            # 存储修改后的中心
            nearby_centers[center_idx, dim_idx, :] = modified_center
    return nearby_centers

def constraint_loss(hash_feat, hash_centers, labels, direction=0):
    # 首先归一化特征和中心
    feat_norm = F.normalize(hash_feat, dim=-1)
    nearby_centers = find_nearby_centers(hash_centers)[:, direction, :].cuda()  # [num_centers, dim, dim] -> [num_centers, dim]
    nearby_centers_norm = F.normalize(nearby_centers, dim=-1)

    # 得到每个hash_feat对应的nearby_centers_norm
    nearby_centers_norm_label = nearby_centers_norm[labels]

    # feat_norm [batch_size, dim], nearby_centers_norm_label [batch_size, dim]
    cos_sim = torch.mul(feat_norm, nearby_centers_norm_label).sum(dim=1)
    loss = cos_sim.mean()

    return loss

def diff_loss(protop_centers, samples_per_class = 10, L = 12, dis = 3, alpha=0.95):
    labels = torch.arange(protop_centers.shape[0]) // samples_per_class
    dot_product = torch.matmul(protop_centers, protop_centers.T)
    hamming_distance = 0.5 * (L*alpha - dot_product)
    mask_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    mask_diff = mask_diff.cuda()
    loss_diff = (F.relu(dis - hamming_distance) * mask_diff.float()).sum(-1)
    return loss_diff.mean()

def same_loss(protop_centers, num_classes = 100, samples_per_class = 10):
    losses = []
    for i in range(num_classes):
        start_idx = i * samples_per_class
        end_idx = (i + 1) * samples_per_class
        class_samples = protop_centers[start_idx:end_idx]
        product = class_samples.unsqueeze(0) * class_samples.unsqueeze(1)
        class_loss = F.relu(-product)
        losses.append(class_loss.sum(dim=-1))
    loss_tensor = torch.cat(losses, dim=0)
    return loss_tensor.mean()

# def hamming_distance_statistics(hash_centers_positive):
#     logger = logging.getLogger("validate")
#     # logger.info("Start validation")

#     # 计算hash_centers_positive的形状
#     num_centers, dim = hash_centers_positive.shape
    
#     # 初始化一个字典来存储各个汉明距离的计数
#     distance_counts = {i: 0 for i in range(dim + 1)}
    
#     # 计算每对中心之间的汉明距离
#     for i in range(num_centers):
#         for j in range(i + 1, num_centers):
#             # 计算两个中心之间的汉明距离
#             distance = torch.sum(hash_centers_positive[i] != hash_centers_positive[j]).item()
#             # 更新对应汉明距离的计数
#             distance_counts[distance] += 1
    
#     # 打印各个汉明距离的计数
#     for distance, count in distance_counts.items():
#         logger.info(f"Hamming distance {distance}: {count}")

def mix_samples(samples, targets):
    # 确定batch大小和设备
    batch_size = samples.size(0)
    device = samples.device
    
    # 创建新的samples和targets容器
    mixed_samples = torch.zeros_like(samples)
    mixed_targets = torch.zeros(batch_size, dtype=torch.long)
    
    # 创建标签映射，第0类和第99类混合产生第100类，依此类推
    label_mapping = {i: 100 + i for i in range(50)}
    label_mapping.update({99 - i: 100 + i for i in range(50)})
    
    # 遍历batch中的每个样本
    for i in range(batch_size):
        # 获取当前样本的标签
        current_label = targets[i].item()
        
        # 查找与当前标签配对的标签
        if current_label in label_mapping:
            mixed_label = label_mapping[current_label]
            # 寻找配对标签的索引
            paired_index = (targets == (99 - current_label)).nonzero(as_tuple=True)[0]
            if len(paired_index) > 0:
                # 如果找到配对，混合当前样本和配对样本
                paired_sample = samples[paired_index[0]]
                mixed_samples[i] = (samples[i] + paired_sample) / 2
                mixed_targets[i] = mixed_label
            else:
                # 如果没有找到配对，只复制当前样本和标签
                mixed_samples[i] = samples[i]
                mixed_targets[i] = current_label
        else:
            # 如果当前标签没有配对项，也只复制当前样本和标签
            mixed_samples[i] = samples[i]
            mixed_targets[i] = current_label
    
    return mixed_samples, mixed_targets

def calculate_alpha(epoch, max_epoch=100, max_alpha=0.5):
    if epoch < max_epoch:
        return (max_alpha / max_epoch) * epoch
    else:
        return max_alpha


def train_one_epoch(model: torch.nn.Module, criterion: _Loss,
                    data_loader: Iterable, data_loader_val, test_loader_unlabelled,
                    optimizer: torch.optim.Optimizer,
                    tb_writer: None, iteration: int,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    args=None,
                    set_training_mode=True,):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 30

    logger = logging.getLogger("train")
    logger.info("Start train one epoch")
    it = 0
    sup_con_crit = SupConLoss()
    ## Rankstat
    # all_combinations = list(combinations(range(0, 32), 3))
    # random.seed(42)  
    # selected_combinations = random.sample(all_combinations, 100)
    # # 建立字典，键为从0开始的索引，值为选取的组合
    # combination_dict = {i: list(comb) for i, comb in enumerate(selected_combinations)}
    # # print(combination_dict)
    # # sys.exit(1)
    ## hash
    # hash_dict = generate_hash_values(length=12, seed=0)
    ## WTA
    # all_combinations = list(product(range(16), repeat=3))
    # random.seed(42)
    # selected_combinations = random.sample(all_combinations, 100)
    # combination_dict = {i: list(comb) for i, comb in enumerate(selected_combinations)}
    # infront_100 = all_combinations[:100]
    combination_dict = {i: list(comb) for i, comb in enumerate(improved_points)}
    # 初始化交叉熵损失函数
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = SoftTargetCrossEntropy()

    # Y = torch.randn(len(data_loader.dataset), 100).float().cuda()
    # U = torch.randn(len(data_loader.dataset), 12).cuda()

    # for samples, targets, _, ind in metric_logger.log_every(data_loader, print_freq, header):
    for batch_index, (samples, targets, _, ind) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # print("samples[0].shape", samples[0].shape)
        # print("samples[1].shape", samples[1].shape)
        samples = samples.to(device, non_blocking=True)
        # weak_samples = samples[0].to(device, non_blocking=True)
        # strong_samples = samples[1].to(device, non_blocking=True)
        # samples = torch.cat(samples, dim=0).to(device)
        targets = targets.to(device, non_blocking=True)
        # targets = torch.cat([targets, targets], dim=0).to(device)
        # targets_onehot = torch.nn.functional.one_hot(targets, num_classes=98).to(device, non_blocking=True)
        # print(targets)
        # print(targets.shape)
        # mixed_samples, mixed_targets = mixup_data(samples, targets_onehot, alpha=2.0)

        # samples = torch.cat((samples, mixed_samples), dim=0)
        # targets_onehot = torch.cat((targets_onehot, mixed_targets), dim=0)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # print("samples.shape:", samples.shape)
            outputs, hash_feat, global_activations, feat = model(samples)
            # outputs = outputs.chunk(2)[0]
            # hash_feat = hash_feat.chunk(2)[0]
            # outputs, _, _ = model(weak_samples)
            # _, hash_feat, _ = model(samples)

            ##pair loss
            # targets_onehot = F.one_hot(targets, num_classes=100).float().cuda()
            # loss_pair = pair_loss(hash_feat, targets_onehot, U, Y)


            # U[ind, :] = hash_feat.data.float()
            # Y[ind, :] = targets_onehot.data
            ##pair loss
            # loss_protop = criterion(outputs, torch.cat([targets, targets], dim=0))
            # loss_protop = loss_function(outputs, targets_onehot)
            ##neg?
            outputs = F.log_softmax(outputs, dim=1)
            # print("outputs.max:", outputs.max())
            # print("outputs.min:", outputs.min())
            loss_protop = torch.nn.NLLLoss()(outputs, targets)
            ##neg?
            # loss_backbone = criterion(outputs_backbone, torch.cat([targets, targets], dim=0))

            ##mixup loss
            # mixed_global_activations, targets_a, targets_b, lam = mixup_data(global_activations, targets, alpha=1.0, use_cuda=True)
            # mixed_feat = model.rankstat_head(mixed_global_activations)
            # feat_mixed = lam * feat[targets_a] + (1 - lam) * feat[targets_b]
            # mse_loss = torch.nn.MSELoss()
            # loss_mix = mse_loss(mixed_feat, feat_mixed)

            #mixupnew
            # output_tensor = swap_topk_values(global_activations.detach().clone(), class_num=100)
            # mixed_global_activations, targets_a, targets_b, lam = mixup_data_ab(global_activations, output_tensor, targets)
            # mixed_feat = model.rankstat_head(mixed_global_activations)

            # feat_mixed = lam * feat[targets_a] + (1 - lam) * model.rankstat_head(output_tensor)[targets_b]
            # mse_loss = torch.nn.MSELoss()
            # loss_mix = mse_loss(mixed_feat, feat_mixed)

            # print(loss_mix)

            ## RankStat
            # comb_indices = torch.tensor([combination_dict[i] for i in targets.tolist()], dtype=torch.long, device=feat.device)
            # # print('comb_indices.shape', comb_indices.shape)
            # # print('combination_dict.keys()', combination_dict.keys())
            # # print('targets.max()', targets.max())
            # # print('targets', targets)
            # # sys.exit(1)
            # # 初始化一个mask，其形状为[batch_size, feat_dim]，feat_dim为特征维度
            # mask = torch.zeros_like(feat, dtype=torch.bool)
            # # 构建适用于scatter_的索引张量
            # batch_size, feat_dim = feat.shape
            # batch_indices = torch.arange(batch_size)[:, None].expand(-1, comb_indices.shape[1])
            # # 使用scatter_填充mask
            # mask[batch_indices, comb_indices] = True
            # # 计算组合对应位置的数值之和
            # comb_sum = (feat * mask).sum(1)
            # # 计算组合外的位置的数值之和
            # # total_sum = (feat * ~mask).sum(1)
            # # 计算损失：组合外位置数值之和减去组合对应位置数值之和
            # # loss_rs = (total_sum - comb_sum).mean()
            # loss_rs = (-comb_sum).mean()

            # f1, f2 = [f for f in feat.chunk(2)]
            # sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # sup_con_labels = targets
            # loss_con = sup_con_crit(sup_con_feats, labels=sup_con_labels)

            ## hash
            # hash_labels_list = [hash_dict[label.item()] for label in targets]
            # hash_labels = torch.tensor(hash_labels_list).float().to(device)
            # loss_hash = criterion(feat, hash_labels)
            ##WTA
            # comb_indices = torch.tensor([combination_dict[i] for i in targets.tolist()], dtype=torch.long, device=feat.device)
            # feat = feat.view(-1, 3, 16)
            # feat = torch.nn.functional.normalize(feat, dim=-1)

            # # 为每个位置初始化一个损失列表
            # losses = []
            # # 遍历comb_indices的每一列
            # for i in range(3):
            #     # 提取当前列的标签
            #     labels = comb_indices[:, i]
            #     predictions = feat[:, i, :]
            #     # 计算交叉熵损失
            #     loss = loss_function(predictions, labels)
            #     # 将当前列的损失添加到损失列表中
            #     losses.append(loss)

            # loss_entropy = sum(losses) / len(losses)

            # batch_size = feat.shape[0]
            # batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, comb_indices.shape[-1])
            # mask = torch.zeros_like(feat, dtype=torch.bool)
            # mask[batch_indices, torch.arange(comb_indices.shape[-1]).unsqueeze(0), comb_indices] = True
            # comb_sum = (feat * mask).sum(-1).sum(-1)
            # loss_rs = (-comb_sum).mean()

            # loss_reg = torch.mean(feat * ~mask, dim=-1).mean()

            ##Openset
            # openset_logits = openset_logits.view(openset_logits.size(0), 2, -1)
            # open_loss_pos, open_loss_neg = ova_loss(openset_logits, targets)
            # loss_os = 0.5 * (open_loss_pos + open_loss_neg)

            # loss = loss_protop + loss_backbone + loss_rs * 0.25 + loss_con * 0.75

            ## hash center loss
            samples_per_class = 10
            class_means = torch.stack([model.prototype_vectors_global[i:i+samples_per_class].mean(0) for i in range(0, model.prototype_vectors_global.size(0), samples_per_class)])
            hash_centers = model.hash_head(class_means)
            # print("hash_centers.shape", hash_centers.shape)
            # print("model.prototype_vectors_global", model.prototype_vectors_global.shape)
            hash_centers_sign = torch.nn.Tanh()(hash_centers*3)
            hash_feat_sign = torch.nn.Tanh()(hash_feat*3)
            # ## print hash center nums
            hash_centers_positive = hash_centers > 0
            # hash_centers_positive_int = hash_centers_positive.int().cpu().numpy()
            # unique_rows = set(tuple(row) for row in hash_centers_positive_int)
            # print(f"Nums of hash centers: {len(unique_rows)}")

            # print(hash_centers.shape)
            loss_diff = diff_loss(hash_centers_sign, samples_per_class=1, L=12, dis=3)
            # loss_same = same_loss(hash_centers, num_classes=100, samples_per_class=10)
            loss_quan = (1 - torch.abs(hash_centers_sign)).mean() 
            ## hash center loss
            # loss_quan2 = (0.5 - torch.abs(hash_feat)).pow(2).mean()
            loss_quan2 = (1 - torch.abs(hash_feat)).mean()
            loss_quan3 = (1 - torch.abs(hash_feat_sign)).mean()


            ## hash loss
            hash_centers_fix = torch.tensor(selected_combinations).float().cuda()
            loss_centers = cos_eps_loss(hash_feat, targets, hash_centers) #+ (1 - torch.abs(hash_feat)).mean() * 0.1
            # loss_centers = center_loss(hash_feat,targets, hash_centers) #+ (1 - torch.abs(hash_feat)).mean() * 0.1
            # loss_centers = cos_eps_loss(u=hash_feat, y=targets, hash_center=torch.Tensor(np.array(selected_combinations)).cuda())

            # loss_reg = reg_loss(hash_feat, targets, hash_centers) #+ (1 - torch.abs(hash_feat)).mean()

            # loss_centers = center_loss_new(hash_feat, targets, hash_centers)


            # loss_reg = reg_loss(hash_feat, targets, hash_centers)
            # loss_quan = (1 - torch.abs(hash_feat)).mean()
            # loss_hash = torch.nn.CrossEntropyLoss()(hash_feat, hash_centers[targets])
            # feat_norm = F.normalize(hash_feat, dim=-1)
            # loss_hash = -torch.sum(torch.mul(feat_norm, hash_centers[targets]), dim=-1).mean()
            # print(loss_hash)

            # contrastive_logits, contrastive_labels = info_nce_logits(feat, device='cuda', n_views=2, negative_mixup=True)
            # loss_con = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
            # alpha = calculate_alpha(epoch)
            
            loss = loss_protop * 1.0 + loss_diff * 0.2 + loss_quan * 0.2 + loss_centers * 3.0 #+ loss_con*alpha#+ loss_quan2 * 0.2#+ loss_centers * 1.0 #+ loss_quan * 1.0#+ loss_quan2 * 1.0
            # loss = loss_protop * 1.0 + loss_diff * 0.1 + loss_quan * 0.1
            # if epoch > 2:
            #     loss = loss_protop * 1.0 + loss_diff * 0.5 + loss_quan * 0.5 + loss_centers * 0.25 + loss_quan2 * 0.5
            #     loss = loss_protop * 1.0 + loss_centers * 0.1 + loss_diff * 0.1
            

            # loss_cons = constraint_loss(hash_feat, hash_centers, targets, direction=epoch%12)
            # print("direction=", batch_index%12)

            # loss = loss_rs + 1.5 * loss_reg
            # loss = loss_entropy
            # if epoch > 20:
            #     loss += loss_centers * 2.0
            #     loss += loss_diff * 0.1
                # loss += loss_same
                # loss += loss_quan 
            # loss += loss_centers
                # print(outputs)
                # print("outputs.max()", outputs.max())
                # print("outputs.min", outputs.min)
                # loss += loss_reg * 2.5
                # loss += loss_pair
            # if epoch > 100:
                # loss += loss_cons
            #     loss += loss_pair
                # loss += loss_centers 
                # loss += loss_quan*0.1
                # loss += loss_os
            # outputs, features = model(samples)

            # loss_cls = criterion(outputs, torch.cat([targets, targets], dim=0))

            # features = torch.nn.functional.normalize(features, dim=-1)

            # f1, f2 = [f for f in features.chunk(2)]
            # sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # sup_con_labels = targets
            # loss_con = sup_con_crit(sup_con_feats, labels=sup_con_labels)

            # if args.use_ppc_loss:
            #     ppc_cov_coe, ppc_mean_coe = args.ppc_cov_coe, args.ppc_mean_coe
            #     total_proto_act, cls_attn_rollout, original_fea_len = auxi_item[2], auxi_item[3], auxi_item[4]
            #     if hasattr(model, 'module'):
            #         ppc_cov_loss, ppc_mean_loss = model.module.get_PPC_loss(total_proto_act, cls_attn_rollout, original_fea_len, targets)
            #     else:
            #         ppc_cov_loss, ppc_mean_loss = model.get_PPC_loss(total_proto_act, cls_attn_rollout, original_fea_len, targets)

            #     ppc_cov_loss = ppc_cov_coe * ppc_cov_loss
            #     ppc_mean_loss = ppc_mean_coe * ppc_mean_loss
            #     if epoch >= 20:
            #         loss = loss + ppc_cov_loss + ppc_mean_loss
            # loss = 0.5*loss_cls+0.5*loss_con
        # loss_value = loss.item()
        # loss_protop_value = loss_protop.item()
        # loss_backbone_value = loss_backbone.item()
        # loss_rs_value = loss_rs.item()
        # loss_con_value = loss_con.item()

        # if not math.isfinite(loss_value):
        # if not math.isfinite(loss_protop_value):
        #     # logger.info("Loss is {}, stopping training".format(loss_value))
        #     logger.info("Loss is {}, stopping training".format(loss_protop_value))
        #     sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # metric_logger.update(loss=loss_value)
        metric_logger.update(loss_protop=loss_protop.item())
        metric_logger.update(loss_centers=loss_centers.item())
        metric_logger.update(loss_diff=loss_diff.item())
        
        metric_logger.update(loss_quan=loss_quan.item())
        metric_logger.update(loss_quan2=loss_quan2.item())
        # metric_logger.update(loss_con=loss_con.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # tb_writer.add_scalars(
        #     main_tag="train/loss",
        #     tag_scalar_dict={
        #         "cls": loss.item(),
        #     },
        #     global_step=iteration+it
        # )
        # if args.use_global and args.use_ppc_loss:
        #     tb_writer.add_scalars(
        #         main_tag="train/ppc_cov_loss",
        #         tag_scalar_dict={
        #             "ppc_cov_loss": ppc_cov_loss.item(),
        #         },
        #         global_step=iteration+it
        #     )
        #     tb_writer.add_scalars(
        #         main_tag="train/ppc_mean_loss",
        #         tag_scalar_dict={
        #             "ppc_mean_loss": ppc_mean_loss.item(),
        #         },
        #         global_step=iteration+it
        #     )
        it += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    hamming_distance_statistics(hash_centers_positive)

    evaluate(data_loader=data_loader_val, test_loader_unlabelled=test_loader_unlabelled, model=model, device=device, args=args, centers=hash_centers.cpu().sign())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def get_img_mask(data_loader, model, device, args):
    logger = logging.getLogger("get mask")
    logger.info("Get mask")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Get Mask:'

    # switch to evaluation mode
    model.eval()

    all_attn_mask = []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            cat_mask = model.get_attn_mask(images)
            all_attn_mask.append(cat_mask.cpu())
    all_attn_mask = torch.cat(all_attn_mask, dim=0) # (num, 2, 14, 14)
    if hasattr(model, 'module'):
        model.module.all_attn_mask = all_attn_mask
    else:
        model.all_attn_mask = all_attn_mask


def compute_hamming_distance_list(list1, list2):
    """计算两个列表之间的汉明距离"""
    # 使用列表推导式和zip，比较对应元素是否不同
    differences = [x != y for x, y in zip(list1, list2)]
    # 计算不同元素的数量，即汉明距离
    hamming_distance = sum(differences)
    return hamming_distance

@torch.no_grad()
def evaluate(data_loader, test_loader_unlabelled, model, device, args, centers):
    logger = logging.getLogger("validate")
    logger.info("Start validation")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    # all_token_attn, pred_labels = [], []
    ##
    # all_feats = []
    # targets = []
    # ##
    # for images, target, _, _ in metric_logger.log_every(data_loader, 10, header):
    #     images = images.to(device, non_blocking=True)
    #     target = target.to(device, non_blocking=True)

    #     # compute output
    #     with torch.cuda.amp.autocast():
    #         ##
    #         # output, auxi_items = model(images,)
    #         output_protop, auxi_items, feat = model(images,)
    #         feat = torch.nn.functional.normalize(feat, dim=-1)[:, :]
    #         ##
    #         # loss = criterion(output, target)
    #         # loss_protop = criterion(output_protop, target)
    #         # loss_backbone = criterion(output_backbone, target)
    #         ##
    #         all_feats.append(feat.cpu().numpy())
    #         targets.append(target.cpu().numpy())
    #         ##

    #     acc1_protop = accuracy(output_protop, target)[0]
    #     # acc1_backbone = accuracy(output_backbone, target)[0]
    #     # acc1, acc5 = accuracy(loss_protop, target, topk=(1, 5))
    #     # _, pred = loss_protop.topk(k=1, dim=1)
    #     # pred_labels.append(pred)

    #     batch_size = images.shape[0]
    #     # metric_logger.update(loss=loss_protop.item())
    #     metric_logger.meters['acc1_protop'].update(acc1_protop.item(), n=batch_size)
    #     # metric_logger.meters['acc1_backbone'].update(acc1_backbone.item(), n=batch_size)

    #     if args.use_global:
    #         # global_acc1 = accuracy(auxi_items[2], target)[0]
    #         local_acc1 = accuracy(auxi_items[3], target)[0]
    #         # metric_logger.meters['global_acc1'].update(global_acc1.item(), n=batch_size)
    #         metric_logger.meters['local_acc1'].update(local_acc1.item(), n=batch_size)
    #     # all_token_attn.append(auxi_items[0])
    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()

    # ## Rankstat
    # all_feats = np.concatenate(all_feats, axis=0)
    # targets = np.concatenate(targets, axis=0)

    # ##RS
    # # top_3_indices = np.argsort(-all_feats, axis=1)[:, :3]
    # # sorted_top_3_indices = np.sort(top_3_indices, axis=1).tolist()
    # ##Hash
    # feats_hash = torch.Tensor(all_feats > 0).float().tolist()
    # ##WTA
    # # all_feats = all_feats.reshape(-1, 3, 16)
    # # sorted_top_3_indices = np.argsort(-all_feats, axis=-1)[:,:,0]
    # # print(sorted_top_3_indices)

    # preds = []
    # hash_dict = []
    # for feat in feats_hash:
    #     if not feat in hash_dict:
    #         hash_dict.append(feat)
    #     preds.append(hash_dict.index(feat))
    # preds = np.array(preds)
    # acc = evaluate_accuracy(preds, targets)
    # logger.info(f'len(list(set(preds))): {len(list(set(preds)))} len(preds): {len(preds)}')
    # logger.info(f"hash Acc: {acc:.3f}")

    # # n_clusters = 100
    # # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # # kmeans.fit(all_feats)

    # # kmeans_labels = kmeans.labels_
    # # acc = evaluate_accuracy(kmeans_labels, targets)

    # # logger.info('* Acc1_protop {top1.global_avg:.3f} Acc1_backbone {top5.global_avg:.3f}'
    # #       .format(top1=metric_logger.acc1_protop, top5=metric_logger.acc1_backbone))
    # logger.info('* Acc1_protop {top1.global_avg:.3f}'.format(top1=metric_logger.acc1_protop))
    # # logger.info(f"Sup con acc: {acc:.3f}")

    ## test_dataset_unlabelled
    all_feats = []
    targets = np.array([])
    mask = np.array([])
    # total_pred_old = 0  # 预测为旧类别的总数
    # correct_pred_old = 0  # 正确预测为旧类别的数量
    # predicted_in_top_100 = []
    # test_logits_local = []
    for batch_idx, (images, label, _, _) in enumerate(tqdm(test_loader_unlabelled)):
        images = images.cuda()
        label = label.cuda()
        logits_local, _, feats = model(images)

        ## openset
        # pred = logits_local.data.max(1)[1]
        # openset_logits = F.softmax(openset_logits.view(openset_logits.size(0), 2, -1), 1)
        # batch_indices = torch.arange(openset_logits.size(0)).to('cuda')
        # selected_values = openset_logits[batch_indices, 1, pred]

        # threshold = 0.65
        # pred_old_class = selected_values > threshold
        # total_pred_old += pred_old_class.sum().item()
        # old_class_mask = label < 100
        # correct_pred_old += (pred_old_class & old_class_mask).sum().item()
        # predicted_in_top_100.append(pred_old_class.cpu().numpy())
        # test_logits_local.append(logits_local.cpu())
        ## openset
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        logger.info(f"labeled_nums: {args.labeled_nums}")
        mask = np.append(mask, np.array([True if x.item() in range(args.labeled_nums) else False for x in label]))

    # if total_pred_old > 0:ß
    #     correct_ratio = correct_pred_old / total_pred_old
    #     logger.info(f"Total old: {total_pred_old}, Correct old: {correct_pred_old}, Correct ratio: {correct_ratio:.2f}")
    # else:
    #     logger.info(f"Total old: {total_pred_old}, Correct old: {correct_pred_old}, Correct ratio: N/A")

    # predicted_in_top_100 = np.concatenate(predicted_in_top_100, axis=0)
    # test_logits_local = torch.cat(test_logits_local, dim=0)
    
    all_feats = np.concatenate(all_feats, axis=0)

    ##RS
    # top_3_indices = np.argsort(-all_feats, axis=1)[:, :3]
    # sorted_top_3_indices = np.sort(top_3_indices, axis=1).tolist()
    ##Hash
    feats_hash = torch.Tensor(all_feats > 0).float().tolist()
    ##WTA
    # all_feats = all_feats.reshape(-1, 3, 16)
    # sorted_indices = np.argsort(-all_feats, axis=-1)
    # sorted_top_3_indices = sorted_indices[:,:,0]

    # preds = []
    # hash_dict = []
    # for feat in feats_hash:
    #     if not feat in hash_dict:
    #         hash_dict.append(feat)
    #     preds.append(hash_dict.index(feat))
    # preds = np.array(preds)
    # all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets, y_pred=preds, mask=mask)
    # logger.info(f'test len(list(set(preds))): {len(list(set(preds)))} len(preds): {len(preds)}')
    # logger.info(f"test hash all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")

    # np.save('car_targets_ocdtransform_ocdtransform_3e-4.npy', targets)
    # np.save('car_hash_feats_ocdtransform_ocdtransform_3e-4.npy', all_feats)
    # np.save('car_hash_centers_ocdtransform_ocdtransform_3e-4.npy', centers.numpy())

    ##labelcluster
    # hash_dict = [[point] for point in improved_points]  # 存储类别的列表，每个类别自身也是一个列表，存储属于该类别的feat
    # hash_dict = [[center] for center in centers.numpy().tolist()] 
    hash_dict = centers.numpy().tolist()
    preds1 = []  # 存储每个feat对应的类别索引
    for feat in feats_hash:
        found = False
        # 首先检查是否已经存在相同的类别索引
        if feat in hash_dict:
            preds1.append(hash_dict.index(feat))  # 使用该类别的索引
            found = True
            
        if not found:
            # 如果没有找到相同的类别索引，再按距离判断
            distances = [compute_hamming_distance_list(feat, center) for center in hash_dict]
            min_distance = min(distances)
            min_index = distances.index(min_distance)

            if min_distance <= 1:
                preds1.append(min_index)
                found = True

        if not found:
            # 如果feat与所有已有类别的距离都大于1，则创建一个新的类别
            hash_dict.append(feat)  # 直接添加整个feat，而不是仅top3_index
            preds1.append(len(hash_dict) - 1)  # 使用新类别的索引
    preds1 = np.array(preds1)

    all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets, y_pred=preds1, mask=mask)
    logger.info(f'test len(list(set(preds1))): {len(list(set(preds1)))} len(preds): {len(preds1)}')
    logger.info(f"case 1 all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")

    # all_acc, old_acc, new_acc = split_cluster_acc_v1(y_true=targets, y_pred=preds1, mask=mask)
    # logger.info(f"case 1 V1_all_acc: {all_acc:.3f} V1_old_acc: {old_acc:.3f} V1_new_acc: {new_acc:.3f}")

    # if all_acc > 0.36:
    #     np.save('cub_targets_36.npy', targets)
    #     np.save('car_hash_feats_36.npy', all_feats)
    #     np.save('car_hash_centers_36.npy', centers.numpy())         


    # hash_dict = centers.numpy().tolist()
    # preds1 = []  # 存储每个feat对应的类别索引
    # for feat in feats_hash:
    #     found = False
    #     # 首先检查是否已经存在相同的类别索引
    #     if feat in hash_dict:
    #         preds1.append(hash_dict.index(feat))  # 使用该类别的索引
    #         found = True
            
    #     if not found:
    #         # 如果没有找到相同的类别索引，再按距离判断
    #         distances = [compute_hamming_distance_list(feat, center) for center in hash_dict]
    #         min_distance = min(distances)
    #         min_index = distances.index(min_distance)

    #         if min_distance <= 2:
    #             preds1.append(min_index)
    #             found = True

    #     if not found:
    #         # 如果feat与所有已有类别的距离都大于1，则创建一个新的类别
    #         hash_dict.append(feat)  # 直接添加整个feat，而不是仅top3_index
    #         preds1.append(len(hash_dict) - 1)  # 使用新类别的索引
    # preds1 = np.array(preds1)

    # all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets, y_pred=preds1, mask=mask)
    # logger.info(f'test len(list(set(preds1))): {len(list(set(preds1)))} len(preds): {len(preds1)}')
    # logger.info(f"case 2 all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")

    # all_acc, old_acc, new_acc = split_cluster_acc_v1(y_true=targets, y_pred=preds1, mask=mask)
    # logger.info(f"case 2 V1_all_acc: {all_acc:.3f} V1_old_acc: {old_acc:.3f} V1_new_acc: {new_acc:.3f}")


    # hash_dict = centers.numpy().tolist()
    # preds1 = []  # 存储每个feat对应的类别索引
    # for feat in feats_hash:
    #     found = False
    #     # 首先检查是否已经存在相同的类别索引
    #     if feat in hash_dict:
    #         preds1.append(hash_dict.index(feat))  # 使用该类别的索引
    #         found = True
            
    #     if not found:
    #         # 如果没有找到相同的类别索引，再按距离判断
    #         distances = [compute_hamming_distance_list(feat, center) for center in hash_dict]
    #         min_distance = min(distances)
    #         min_index = distances.index(min_distance)

    #         if min_distance <= 3:
    #             preds1.append(min_index)
    #             found = True

    #     if not found:
    #         # 如果feat与所有已有类别的距离都大于1，则创建一个新的类别
    #         hash_dict.append(feat)  # 直接添加整个feat，而不是仅top3_index
    #         preds1.append(len(hash_dict) - 1)  # 使用新类别的索引
    # preds1 = np.array(preds1)

    # all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets, y_pred=preds1, mask=mask)
    # logger.info(f'test len(list(set(preds1))): {len(list(set(preds1)))} len(preds): {len(preds1)}')
    # logger.info(f"case 3 all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")

    # all_acc, old_acc, new_acc = split_cluster_acc_v1(y_true=targets, y_pred=preds1, mask=mask)
    # logger.info(f"case 3 V1_all_acc: {all_acc:.3f} V1_old_acc: {old_acc:.3f} V1_new_acc: {new_acc:.3f}")

    # hash_dict = centers.numpy().tolist()
    # preds1 = []  # 存储每个feat对应的类别索引
    # for feat in feats_hash:
    #     found = False
    #     # 首先检查是否已经存在相同的类别索引
    #     if feat in hash_dict:
    #         preds1.append(hash_dict.index(feat))  # 使用该类别的索引
    #         found = True
            
    #     if not found:
    #         # 如果没有找到相同的类别索引，再按距离判断
    #         distances = [compute_hamming_distance_list(feat, center) for center in hash_dict]
    #         min_distance = min(distances)
    #         min_index = distances.index(min_distance)

    #         if min_distance <= 2:
    #             preds1.append(min_index)
    #             found = True

    #     if not found:
    #         # 如果feat与所有已有类别的距离都大于1，则创建一个新的类别
    #         hash_dict.append(feat)  # 直接添加整个feat，而不是仅top3_index
    #         preds1.append(len(hash_dict) - 1)  # 使用新类别的索引
    # preds1 = np.array(preds1)

    # all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets, y_pred=preds1, mask=mask)
    # logger.info(f'test len(list(set(preds1))): {len(list(set(preds1)))} len(preds): {len(preds1)}')
    # logger.info(f"case 3 all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")

    # hash_dict = []
    # preds = []  # 存储每个feat对应的类别索引
    # for feat in feats_hash:
    #     found = False
    #     # 首先检查是否已经存在相同的类别索引
    #     if feat in hash_dict:
    #         preds.append(hash_dict.index(feat))  # 使用该类别的索引
    #         found = True
            
    #     if not found:
    #         if hash_dict:
    #             # 如果没有找到相同的类别索引，再按距离判断
    #             distances = [compute_hamming_distance_list(feat, center) for center in hash_dict]
    #             min_distance = min(distances)
    #             min_index = distances.index(min_distance)

    #             if min_distance <= 1:
    #                 preds.append(min_index)
    #                 found = True

    #     if not found:
    #         # 如果feat与所有已有类别的距离都大于1，则创建一个新的类别
    #         hash_dict.append(feat)  # 直接添加整个feat，而不是仅top3_index
    #         preds.append(len(hash_dict) - 1)  # 使用新类别的索引
    # preds = np.array(preds)

    # all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets, y_pred=preds, mask=mask)
    # logger.info(f'test len(list(set(preds))): {len(list(set(preds)))} len(preds): {len(preds)}')
    # logger.info(f"case 2 all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")


    # preds3 = []
    # hash_dict = centers.numpy().tolist()
    # for feat in feats_hash:
    #     if not feat in hash_dict:
    #         hash_dict.append(feat)
    #     preds3.append(hash_dict.index(feat))
    # preds3 = np.array(preds3)
    # all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets, y_pred=preds3, mask=mask)
    # logger.info(f'test len(list(set(preds3))): {len(list(set(preds3)))} len(preds): {len(preds3)}')
    # logger.info(f"case 3 all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")

    # ## case2
    # hash_dict = centers.numpy().tolist()
    # preds = []  # 存储每个feat对应的类别索引
    # for feat in feats_hash:
    #     found = False
    #     # 首先检查是否已经存在相同的类别索引
    #     if feat in hash_dict:
    #         preds.append(hash_dict.index(feat))  # 使用该类别的索引
    #         found = True
            
    #     if not found:
    #         # 首先处理编号100及之后的中心，它们有优先权
    #         for i, center in enumerate(hash_dict):
    #             if i >= 100:  # 对于编号为100及之后的哈希中心
    #                 distance = compute_hamming_distance_list(feat, center)
    #                 if distance <= 2:  # 如果距离小于等于2
    #                     preds.append(i)  # 归入该中心
    #                     found = True
    #                     break
    #         if found:
    #             continue  # 如果已经归类，继续处理下一个特征

    #         # 接下来处理前100个哈希中心
    #         for i, center in enumerate(hash_dict):
    #             if i < 100:  # 对于前100个哈希中心
    #                 distance = compute_hamming_distance_list(feat, center)
    #                 if distance <= 1:  # 如果距离小于等于1
    #                     preds.append(i)  # 归入该中心
    #                     found = True
    #                     break

    #     if not found:
    #         # 如果feat与所有已有类别的距离都大于1，则创建一个新的类别
    #         hash_dict.append(feat)  # 直接添加整个feat，而不是仅top3_index
    #         preds.append(len(hash_dict) - 1)  # 使用新类别的索引
    # preds = np.array(preds)

    # all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets, y_pred=preds, mask=mask)
    # logger.info(f'test len(list(set(preds))): {len(list(set(preds)))} len(preds): {len(preds)}')
    # logger.info(f"case 2 all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")

    # ## case3
    # hash_dict = centers.numpy().tolist()
    # preds = []  # 存储每个feat对应的类别索引
    # for feat in feats_hash:
    #     found = False
    #     # 首先检查是否已经存在相同的类别索引
    #     if feat in hash_dict:
    #         preds.append(hash_dict.index(feat))  # 使用该类别的索引
    #         found = True
            
    #     if not found:
    #         # 首先处理编号100及之后的中心，它们有优先权
    #         for i, center in enumerate(hash_dict):
    #             if i >= 100:  # 对于编号为100及之后的哈希中心
    #                 distance = compute_hamming_distance_list(feat, center)
    #                 if distance <= 1:  # 如果距离小于等于2
    #                     preds.append(i)  # 归入该中心
    #                     found = True
    #                     break

    #     if not found:
    #         # 如果feat与所有已有类别的距离都大于1，则创建一个新的类别
    #         hash_dict.append(feat)  # 直接添加整个feat，而不是仅top3_index
    #         preds.append(len(hash_dict) - 1)  # 使用新类别的索引
    # preds = np.array(preds)

    # all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets, y_pred=preds, mask=mask)
    # logger.info(f'test len(list(set(preds))): {len(list(set(preds)))} len(preds): {len(preds)}')
    # logger.info(f"case 3 all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")

    # hash_dict = [[center] for center in centers.numpy().tolist()]   # 存储类别的列表，每个类别自身也是一个列表，存储属于该类别的feat
    # preds = []  # 存储每个feat对应的类别索引
    # for feat in feats_hash:
    #     found = False
    #     # top3_index = tuple(feat[:, 0])  # 将NumPy数组转换为元组
    #     # 首先检查是否已经存在相同的类别索引
    #     for idx, category in enumerate(hash_dict):
    #         # if any(top3_index == exist_feat for exist_feat in category):
    #         if feat in category:
    #             preds.append(idx)  # 使用该类别的索引
    #             found = True
    #             break
    #     if not found:
    #         # 如果没有找到相同的类别索引，再按距离判断
    #         for idx, category in enumerate(hash_dict):
    #             if any(compute_hamming_distance_list(exist_feat, feat) <= 1 for exist_feat in category):
    #                 # if idx >= 100:
    #                 #     hash_dict[idx].append(top3_index)
    #                 preds.append(idx)  # 使用该类别的索引
    #                 found = True
    #                 break
    #     if not found:
    #         # 如果feat与所有已有类别的距离都大于1，则创建一个新的类别
    #         hash_dict.append([feat])  # 直接添加整个feat，而不是仅top3_index
    #         preds.append(len(hash_dict) - 1)  # 使用新类别的索引
    # preds = np.array(preds)

    # all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets, y_pred=preds, mask=mask)
    # logger.info(f'test len(list(set(preds))): {len(list(set(preds)))} len(preds): {len(preds)}')
    # logger.info(f"Threshold=1 all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")
    ##openset
    # preds = []
    # hash_dict = []
    # # all_combinations = list(product(range(16), repeat=3))
    # # random.seed(42)
    # # selected_combinations = random.sample(all_combinations, 100)
    # combination_dict = {i: list(comb) for i, comb in enumerate(improved_points)}
    # for idx, top_3_feat in enumerate(sorted_top_3_indices.tolist()):
    #     if predicted_in_top_100[idx]:
    #         set_value = torch.argmax(test_logits_local[idx])
    #         pred_value = combination_dict[set_value.item()]
    #         if pred_value not in hash_dict:
    #             hash_dict.append(pred_value)
    #             preds.append(hash_dict.index(pred_value))
    #         else:
    #             preds.append(hash_dict.index(pred_value))
    #     else:
    #         if not top_3_feat in hash_dict:
    #             hash_dict.append(top_3_feat)
    #             preds.append(hash_dict.index(top_3_feat))  # 只在这里添加
    #         else:
    #             preds.append(hash_dict.index(top_3_feat))

    # preds = np.array(preds)
    # all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets, y_pred=preds, mask=mask)
    # logger.info(f'test len(list(set(preds))): {len(list(set(preds)))} len(preds): {len(preds)}')
    # logger.info(f"test openset all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")

    ##openset+labelcluster
    # hash_dict = [[point] for point in improved_points]  # 存储类别的列表，每个类别自身也是一个列表，存储属于该类别的feat
    # preds = []  # 存储每个feat对应的类别索引
    # for id_feat, feat in enumerate(sorted_top_3_indices.tolist()):
    #     if predicted_in_top_100[id_feat]:
    #         set_value = torch.argmax(test_logits_local[id_feat])
    #         preds.append(set_value.item())
    #     else:
    #         found = False
    #         for idx, category in enumerate(hash_dict):
    #             # 检查feat是否与该类别中任一feat的距离小于等于1
    #             if any(calc_distance(feat, exist_feat) <= 1 for exist_feat in category):
    #                 if idx >= 100:
    #                     hash_dict[idx].append(feat)  # 将feat添加到已有类别中
    #                 preds.append(idx)  # 使用该类别的索引
    #                 found = True
    #                 break
    #         if not found:
    #             # 如果feat与所有已有类别的距离都大于1，则创建一个新的类别
    #             hash_dict.append([feat])
    #             preds.append(len(hash_dict) - 1)  # 使用新类别的索引
    # preds = np.array(preds)
    # all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets, y_pred=preds, mask=mask)
    # logger.info(f'test len(list(set(preds))): {len(list(set(preds)))} len(preds): {len(preds)}')
    # logger.info(f"test openset+labelcluster all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")
    ## test_dataset_unlabelled

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        
# @torch.no_grad()
# def evaluate(data_loader, model, device, args):
#     logger = logging.getLogger("validate")
#     logger.info("Start validation")
#     criterion = torch.nn.CrossEntropyLoss()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     # switch to evaluation mode
#     model.eval()

#     all_token_attn, pred_labels = [], []
#     ##
#     all_feats = []
#     targets = []
#     ##
#     for images, target, _ in metric_logger.log_every(data_loader, 10, header):
#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         with torch.cuda.amp.autocast():
#             ##
#             (output, features), auxi_items = model(images,)
#             ##
#             loss = criterion(output, target)
#             ##
#             all_feats.append(features.cpu().numpy())
#             targets.append(target.cpu().numpy())
#             ##

#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#         _, pred = output.topk(k=1, dim=1)
#         pred_labels.append(pred)

#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

#         if args.use_global:
#             global_acc1 = accuracy(auxi_items[2], target)[0]
#             local_acc1 = accuracy(auxi_items[3], target)[0]
#             metric_logger.meters['global_acc1'].update(global_acc1.item(), n=batch_size)
#             metric_logger.meters['local_acc1'].update(local_acc1.item(), n=batch_size)
#         all_token_attn.append(auxi_items[0])
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()

#     all_feats = np.concatenate(all_feats, axis=0)
#     targets = np.concatenate(targets, axis=0)

#     n_clusters = 100
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     kmeans.fit(all_feats)

#     kmeans_labels = kmeans.labels_
#     acc = evaluate_accuracy(kmeans_labels, targets)

#     logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
#     logger.info(f"Sup con acc: {acc:.3f}")
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc

def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def split_cluster_acc_v1(y_true, y_pred, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    old_acc = cluster_acc(y_true[mask], y_pred[mask])
    new_acc = cluster_acc(y_true[~mask], y_pred[~mask])
    total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc