import torch
import random
import logging
import warnings
import numpy as np
import torch.nn.functional as F

formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def euclidean_d(X, Y=None, squared=True):
    if Y is None:
        Y = X
    # N*1*M
    XX = X.unsqueeze(dim=1)

    # 1*N*M
    YY = Y.unsqueeze(dim=0)

    dis = (XX - YY) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1)
    if squared:
        return dis
    else:
        return torch.sqrt(dis)


def euclidean_d_1(X, Y=None, squared=True):
    if Y is None:
        Y = X
    # N*1*M
    XX = X.unsqueeze(dim=2)

    # 1*N*M
    YY = Y.unsqueeze(dim=1)

    dis = (XX - YY) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1)
    if squared:
        return dis
    else:
        return torch.sqrt(dis)


def kmean_pp(X, dists, n_clusters, index_bias, seed=None):
    seed = seed or 114514
    random_state = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    # centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    centers_indices = np.empty((n_clusters, ), dtype=np.int)
    # n_local_trials = 2 + int(np.log(n_clusters))
    n_local_trials = 1
    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    # weight = normalize(x_squared_norms)
    # center_id = random.choice()
    centers_indices[0] = center_id
    # centers[0] = X[center_id]
    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = dists[center_id]
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        # rand_vals = random_state.random_sample(1) * current_pot
        # candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
        #                                 rand_vals)
        w = closest_dist_sq / current_pot
        candidate_ids = random.choices(range(len(closest_dist_sq)),
                                       w,
                                       k=1)

        # Compute distances to center candidates
        distance_to_candidates = dists[candidate_ids]

        best_candidate = candidate_ids[0]
        best_pot = None
        best_dist_sq = None
        if n_clusters > 2:
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates)
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            best_pot = new_pot
            best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        centers_indices[c] = best_candidate
        current_pot = best_pot
        closest_dist_sq = best_dist_sq
    return centers_indices + index_bias


def pick_samples(preds, label, positive):
    _, pred = preds.topk(1, 1, True, True)
    pred = torch.squeeze(pred.t())
    if positive:
        return torch.where(pred != label)
    else:
        return torch.where(pred == label)


def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def add_filehandler(logger, filepath):
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_model(save_path, model, optimizer=None):
    data = torch.load(save_path)
    if 'model' in data or 'state_dict' in data:
        key = 'model' if 'model' in data else 'state_dict'
        if not isinstance(model, torch.nn.DataParallel):
            model.load_state_dict(
                {k.replace('module.', ''): v for k, v in data[key].items()})
        else:
            model.load_state_dict(
                {k if 'module.' in k else 'module.' + k: v for k, v in
                 data[key].items()})
        if optimizer:
            optimizer.load_state_dict(data['optimizer'])
    else:
        model.load_state_dict({k: v for k, v in data.items()})
    return data


def kl_d(input_logits, target_logits):
    inputs = F.log_softmax(input_logits, 1)
    targets = F.log_softmax(target_logits, 1)
    return torch.sum(torch.exp(inputs) * (inputs - targets), dim=1)


def save_status(epoch, rs, optimizer, model, save_path):
    torch.save({
        'epoch': epoch,
        'log': {
            'train': rs['train'].get_dict(),
            'test': rs['test'].get_dict(),
        },
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict()
    }, save_path)
    torch.save({
        'epoch': epoch,
        'log': {
            'train': rs['train'].get_dict(),
            'test': rs['test'].get_dict(),
        },
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict()
    }, save_path.replace('.pth', '_e%d_top1_%.3f_%.3f' % (epoch, rs['train']['top1'], rs['test']['top1']) + '.pth'))
