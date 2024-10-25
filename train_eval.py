from cProfile import label
import logging
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from torch.nn.modules.loss import _Loss
import tools.utils as utils
# from tensorboardX import SummaryWriter
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from tqdm import tqdm
import math

import torch.nn.functional as F

def cos_eps_loss(u, y, hash_center):

    u_norm = F.normalize(u)
    centers_norm = F.normalize(hash_center)
    cos_sim = torch.matmul(u_norm, torch.transpose(centers_norm, 0, 1)) # batch x n_class

    loss = torch.nn.CrossEntropyLoss()(cos_sim, y)

    return loss

def sep_loss(protop_centers, samples_per_class = 10, L = 12, dis_max = 3, alpha=0.95):
    labels = torch.arange(protop_centers.shape[0]) // samples_per_class
    dot_product = torch.matmul(protop_centers, protop_centers.T)
    hamming_distance = 0.5 * (L*alpha - dot_product)
    mask_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    mask_diff = mask_diff.cuda()
    loss_sep = (F.relu(dis_max - hamming_distance) * mask_diff.float()).sum(-1)
    return loss_sep.mean()



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

    dis_max = get_dis_max(args)

    for batch_index, (samples, targets, _, ind) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)


        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # print("samples.shape:", samples.shape)
            outputs, hash_feat = model(samples)

            # supervise loss
            outputs = F.log_softmax(outputs, dim=1)
            loss_protop = torch.nn.NLLLoss()(outputs, targets)

            ## get hash centers
            samples_per_class = args.global_proto_per_class
            class_means = torch.stack([model.prototype_vectors_global[i:i+samples_per_class].mean(0) for i in range(0, model.prototype_vectors_global.size(0), samples_per_class)])
            hash_centers = model.hash_head(class_means)

            hash_centers_sign = torch.nn.Tanh()(hash_centers*3)

            # hash centers separation loss
            loss_sep = sep_loss(hash_centers_sign, samples_per_class=1, L=args.hash_code_length, dis_max=dis_max)

            # hash center quantization loss
            loss_quan = (1 - torch.abs(hash_centers_sign)).mean() 

            ## hash feature optimize loss
            loss_feature = cos_eps_loss(hash_feat, targets, hash_centers) 
            
            loss = loss_protop * 1.0 + loss_sep * 0.1 + loss_quan * 0.1 + loss_feature * 3.0

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
        metric_logger.update(loss_feature=loss_feature.item())
        metric_logger.update(loss_sep=loss_sep.item())        
        metric_logger.update(loss_quan=loss_quan.item())

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

    evaluate(test_loader_unlabelled=test_loader_unlabelled, model=model, args=args, centers=hash_centers.cpu().sign(), dis_max=dis_max)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def compute_hamming_distance_list(list1, list2):
    """计算两个列表之间的汉明距离"""
    # 使用列表推导式和zip，比较对应元素是否不同
    differences = [x != y for x, y in zip(list1, list2)]
    # 计算不同元素的数量，即汉明距离
    hamming_distance = sum(differences)
    return hamming_distance

@torch.no_grad()
def evaluate(test_loader_unlabelled, model, args, centers, dis_max):
    radius = max(math.floor(dis_max / 2), 1)
    logger = logging.getLogger("validate")
    logger.info("Start validation")

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    for batch_idx, (images, label, _, _) in enumerate(tqdm(test_loader_unlabelled)):
        images = images.cuda()
        label = label.cuda()
        feats = model(images)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        logger.info(f"labeled_nums: {args.labeled_nums}")
        mask = np.append(mask, np.array([True if x.item() in range(args.labeled_nums) else False for x in label]))
    
    all_feats = np.concatenate(all_feats, axis=0)

    ##Hash
    feats_hash = torch.Tensor(all_feats > 0).float().tolist()

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


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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

def get_dis_max(args):
    pass

def get_dis_max(args):

    table = {
        'CD_CUB2011U': {12: 3, 16: 4, 32: 10, 64: 24},
        'CD_Car': {12: 3, 16: 4, 32: 10, 64: 24},
        'CD_pets': {12: 4, 16: 5, 32: 12, 64: 27},
        'CD_food': {12: 3, 16: 5, 32: 11, 64: 25},
        'Fungi': {12: 3, 16: 4, 32: 11, 64: 25},
        'Arachnida': {12: 4, 16: 5, 32: 11, 64: 26},
        'Animalia': {12: 3, 16: 5, 32: 11, 64: 25},
        'Mollusca': {12: 3, 16: 5, 32: 11, 64: 25},
    }
    
    dataset = args.data_set
    hash_code_length = args.hash_code_length
    
    return table.get(dataset, {}).get(hash_code_length, None)