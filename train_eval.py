import math
import torch
import logging
import numpy as np

import torch.nn.functional as F

from tqdm import tqdm

from utils import split_cluster_acc_v1, split_cluster_acc_v2, get_dis_max, SmoothedValue, MetricLogger


def cos_eps_loss(u, y, hash_center):

    u_norm = F.normalize(u)
    centers_norm = F.normalize(hash_center)
    cos_sim = torch.matmul(u_norm, torch.transpose(centers_norm, 0, 1)) # batch x n_class

    loss = torch.nn.CrossEntropyLoss()(cos_sim, y)

    return loss

def sep_loss(protop_centers, L = 12, dis_max = 3, alpha=0.95):
    labels = torch.arange(protop_centers.shape[0])
    dot_product = torch.matmul(protop_centers, protop_centers.T)
    hamming_distance = 0.5 * (L*alpha - dot_product)
    mask_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    mask_diff = mask_diff.cuda()
    loss_sep = (F.relu(dis_max - hamming_distance) * mask_diff.float()).sum(-1)
    return loss_sep.mean()


def train_and_evaluate(model, 
                    data_loader, 
                    test_loader_unlabelled, 
                    optimizer, device, 
                    epoch, loss_scaler, 
                    max_norm, 
                    model_ema, 
                    args=None, 
                    set_training_mode=True):
    
    model.train(set_training_mode)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 30

    logger = logging.getLogger("train")
    logger.info("Start train one epoch")
    it = 0

    dis_max = get_dis_max(args)

    for batch_index, (samples, targets, _, ind) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
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
            loss_sep = sep_loss(hash_centers_sign, L=args.hash_code_length, dis_max=dis_max)

            # hash center quantization loss
            loss_quan = (1 - torch.abs(hash_centers_sign)).mean() 

            ## hash feature optimize loss
            loss_feature = cos_eps_loss(hash_feat, targets, hash_centers) 
            
            loss = loss_protop + loss_sep * args.alpha + loss_quan * args.alpha + loss_feature * args.beta

        optimizer.zero_grad()

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

        it += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    evaluate(test_loader=test_loader_unlabelled, model=model, args=args, centers=hash_centers.cpu().sign(), dis_max=dis_max)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_hamming_distance_list(list1, list2):
    """Compute the Hamming distance between two lists"""
    # Use list comprehension and zip to compare if corresponding elements are different
    differences = [x != y for x, y in zip(list1, list2)]
    # Count the number of differing elements, which is the Hamming distance
    hamming_distance = sum(differences)
    return hamming_distance

@torch.no_grad()
def evaluate(test_loader, model, args, centers, dis_max):
    radius = max(math.floor(dis_max / 2), 1)
    logger = logging.getLogger("validate")
    logger.info("Start validation")
    logger.info(f"Radius: {radius}")

    metric_logger = MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    for batch_idx, (images, label, _, _) in enumerate(tqdm(test_loader)):
        images = images.cuda()
        label = label.cuda()
        feats = model(images)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
 
        mask = np.append(mask, np.array([True if x.item() in range(args.labeled_nums) else False for x in label]))
    
    all_feats = np.concatenate(all_feats, axis=0)

    # Hash
    feats_hash = torch.Tensor(all_feats > 0).float().tolist()

    hash_dict = centers.numpy().tolist()

    # Store the category index corresponding to each feature
    preds1 = []  

    for feat in feats_hash:
        found = False
        # First check if the same category index already exists
        if feat in hash_dict:
            # Use the index of that category
            preds1.append(hash_dict.index(feat))  
            found = True
            
        if not found:
            # If no identical category index is found, then judge based on distance
            distances = [compute_hamming_distance_list(feat, center) for center in hash_dict]
            min_distance = min(distances)
            min_index = distances.index(min_distance)

            if min_distance <= radius:
                preds1.append(min_index)
                found = True

        if not found:
            # If the distance between feat and all existing categories exceeds the Hamming sphere radius, create a new category
            hash_dict.append(feat) 
            # Use the index of the new category as the classification result
            preds1.append(len(hash_dict) - 1)

    preds1 = np.array(preds1)

    all_acc, old_acc, new_acc = split_cluster_acc_v1(y_true=targets, y_pred=preds1, mask=mask)
    logger.info(f"Evaluate V1: all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")

    all_acc, old_acc, new_acc = split_cluster_acc_v2(y_true=targets, y_pred=preds1, mask=mask)
    logger.info(f"Evaluate V2: all_acc: {all_acc:.3f} old_acc: {old_acc:.3f} new_acc: {new_acc:.3f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}