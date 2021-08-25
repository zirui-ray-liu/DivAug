import argparse
import logging
import os
import torchvision as tv
import torch
import time
import math
import yaml
import json
import itertools
import warnings
import torch.nn.functional as F

from tqdm import tqdm
from metrics import Accumulator, accuracy
from collections import OrderedDict
from networks import get_model, get_num_class
from common.utils import set_seed, get_logger, add_filehandler, load_model, save_status
from common.smooth_ce import SoftCrossEntropyLoss
from data import get_dataloaders
from transforms.augment_policy import div_augment
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler
from warmup_scheduler import GradualWarmupScheduler


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Autoaug for the Tencent Project',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-seed', help='the seed used in experiments',
                    default=42, type=int, required=False)

parser.add_argument('-c', '--config', help='the yaml file specifying the configu-ration for training.',
                    required=True)

parser.add_argument('--dataroot', type=str, default='./data',
                    help='torchvision data folder', required=False)

parser.add_argument('--verbose_eval', type=int, default=5,
                    help='evaluate the model every <verbose_eval> epoch', required=False)

parser.add_argument('-exp_name', '--experiment_name',
                    help='the path specified for saving logs, config, and models',
                    default=None, required=False, type=str)

parser.add_argument('--ssl', type=bool, default=False,
                    help='unsupervised', required=False)

parser.add_argument('--only_eval', action='store_true', help='whether to only evaluate the trained model')
parser.add_argument('--save', default='model.pth', help='the saved model name', required=False)


logger = get_logger('Augment')
logger.setLevel(logging.INFO)


def _get_TSA_thresh(config, epoch, steps, steps_per_epoch):
    mode = config.get('ratio_mode', 'linear')
    k = 1 / get_num_class(config['dataset'])
    t_max = config['epoch']
    eta = min((epoch + steps / steps_per_epoch) / t_max, 1)
    if mode == 'constant':
        thresh = 1
    elif mode == 'linear':
        thresh = (1 - k) * eta + k
    elif mode == 'log':
        thresh = (1 - math.exp(-5 * eta)) * (1 - k) + k
    elif mode == 'exp':
        thresh = math.exp(5 * (eta - 1)) * (1 - k) + k
    else:
        raise NotImplementedError
    return thresh


def run_epoch(model, loader_s, loader_u, loss_fn,
              optimizer, config, desc_default='', epoch=0, unsupervised=False,
              writer=None, verbose=True, scheduler=None, scaler=None):
    if verbose:
        loader_s = tqdm(loader_s)
        loader_s.set_description('[%s %04d/%04d]' % (desc_default, epoch, config['epoch']))
    iter_u = iter(loader_u) if loader_u else None
    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader_s)
    steps = 0
    aug = config.get('aug', None)
    for data, label in loader_s:
        steps += 1
        data, label = data.cuda(), label.cuda()
        if not unsupervised:
            if len(data.shape) == 5:
                data = data.flatten(0, 1)
            if optimizer and aug == 'divaug':
                index = div_augment(config['divaug'], data, model)
                label = torch.unsqueeze(label, -1).expand(label.shape[0], config['divaug']['C']).flatten(0, 1)
                data = data[index]
                label = label[index]
            if optimizer:
                optimizer.zero_grad()
            sup_logits = model(data)
            loss = loss_fn(sup_logits, label)
            loss = loss.mean()
        else:
            label = label.cuda()
            label_for_mask = label.type(torch.LongTensor).reshape(-1, 1).cuda()
            try:
                unlabel1, unlabel2 = next(iter_u)
            except StopIteration:
                iter_u = iter(loader_u)
                unlabel1, unlabel2 = next(iter_u)
            unlabel1, unlabel2 = unlabel1.cuda(), unlabel2.cuda()
            if optimizer:
                optimizer.zero_grad()
            if len(unlabel2.shape) == 5:
                unlabel2 = unlabel2.flatten(0, 1)
            if optimizer and aug == 'divaug':
                index = div_augment(config['divaug'], unlabel2, model)
                unlabel2 = unlabel2[index]
            # tv.utils.save_image(data[:16] * 0.2 + 0.5, 'ori_img.png')
            # tv.utils.save_image(unlabel1[:16] * 0.2 + 0.5, 'u1.png')
            # tv.utils.save_image(unlabel2[:32] * 0.2 + 0.5, 'u2.png')
            # raise NotImplementedError
            # data_all = torch.cat([data, unlabel1, unlabel2]).cuda()
            # logits_all = model(data_all)
            # sup_logits = logits_all[:len(data)]
            # ori_logits, aug_logits = logits_all[len(data):len(data)+len(unlabel1)], \
            #                          logits_all[len(data)+len(unlabel1):]
            sup_logits = model(data)
            ori_logits = model(unlabel1)
            del unlabel1
            aug_logits = model(unlabel2)
            del unlabel2

            loss = loss_fn(sup_logits, label)  # loss for supervised learning
            sup_preds = F.softmax(sup_logits, dim=1)
            sup_y_prob = torch.gather(sup_preds, 1, label_for_mask)
            thresh = _get_TSA_thresh(config, epoch, steps, total_steps)
            sup_loss_mask = (sup_y_prob < thresh).squeeze().detach()
            # supervised loss
            loss = (loss * sup_loss_mask).sum() / (torch.sum(sup_loss_mask) + 1e-10)
            ori_preds = F.softmax(ori_logits, dim=1).detach()

            # sharpen the prediction
            ori_logits_sharpened = ori_logits / config.get('softmax_temp', 1.)
            ori_preds_sharpened = F.softmax(ori_logits_sharpened, dim=1).detach()

            # confidence-based masking
            largest_ori_probs = torch.max(ori_preds, dim=1)[0]
            conf_thresh = config.get('confidence_threshold', 0.)
            kl_loss_mask = largest_ori_probs > conf_thresh
            aug_preds = F.log_softmax(aug_logits, dim=1)
            enlarge = config[aug]['S'] if aug == 'divaug' else config[aug].get('C', 1)
            assert len(aug_preds) == len(ori_preds) * enlarge
            aug_preds = aug_preds.reshape(ori_preds.shape[0], enlarge, ori_preds.shape[1])
            loss_kldiv = 0
            for i in range(enlarge):
                cur_loss_kldiv = F.kl_div(aug_preds[:, i, :], ori_preds_sharpened, reduction='none')
                loss_kldiv += torch.sum(cur_loss_kldiv, dim=1)
            loss_kldiv = loss_kldiv / enlarge
            loss_kldiv = loss_kldiv * kl_loss_mask.detach()
            loss += config['ratio_unsup'] * torch.mean(loss_kldiv)

        if optimizer:
            if scaler:
                scaler.scale(loss).backward()
                if config['optimizer'].get('clip', 5) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer'].get('clip', 5))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config['optimizer'].get('clip', 5) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer'].get('clip', 5))
                optimizer.step()

        top1, top5 = accuracy(sup_logits, label, (1, 5))
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
        })
        cnt += len(data)
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            loader_s.set_postfix(postfix)

        if scheduler is not None:
            scheduler.step(epoch - 1 + float(steps) / total_steps)
            # scheduler.step()
        del sup_logits, loss, top1, top5, label, data

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']

    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics


def evaluate(result, rs, metric, best_top1, model, optimizer, testloader,
             criterion, config, epoch, writers, scaler, save_path=True, trainloader=None):

    # if trainloader:
    #     rs['train'] = run_epoch(model, trainloader, None,
    #                             criterion, None, config,
    #                             desc_default='train', epoch=epoch,
    #                             unsupervised=False, writer=writers[0],
    #                             scaler=scaler)
    rs['test'] = run_epoch(model, testloader, None,
                           criterion, None, config,
                           desc_default='*test', epoch=epoch,
                           unsupervised=False, writer=writers[1],
                           scaler=scaler)
    if metric == 'last' or rs[metric]['top1'] > best_top1:
        if metric != 'last':
            best_top1 = rs[metric]['top1']
        else:
            best_top1 = rs['test']['top1']
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['test']):
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = epoch

        writers[1].add_scalar('test_top1/best', rs['test']['top1'], epoch)
    # save checkpoint
    if save_path is not None:
        logger.info('save model@%d to %s' % (epoch, save_path))
        save_status(epoch, rs, optimizer, model, save_path)
    return best_top1


def train_and_eval(args, metric='last'):
    verbose_eval = args.verbose_eval
    config = args.config
    only_eval = True if args.only_eval else False
    save_path = os.path.join('./logs', args.experiment_name, args.save)
    num_class = get_num_class(config['dataset'])
    model = get_model(args.config['model'], args.config['dataset'], num_class)
    # criterion = SoftCrossEntropyLoss(reduction='none')
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if config['optimizer']['type'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['optimizer']['lr'],
            momentum=config['optimizer'].get('momentum', 0.9),
            weight_decay=config['optimizer']['decay'],
            nesterov=config['optimizer']['nesterov']
        )
    else:
        raise ValueError('invalid optimizer type=%s' % config['optimizer']['type'])
    if args.config.get('mixed_precision', False) is True:
        if args.config['model']['type'] not in ['pyramid', 'resnet50', 'wresnet40_2']:
            raise ValueError('Currently only support mixed precision training for PyramidNet and ResNet50')
        scaler = GradScaler()
    else:
        scaler = None

    lr_scheduler_type = config['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        t_max = config['epoch']
        if config['lr_schedule'].get('warmup', None):
            t_max -= config['lr_schedule']['warmup']['epoch']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.)
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

    if config['lr_schedule'].get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=config['lr_schedule']['warmup']['multiplier'],
            total_epoch=config['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )

    writers = [SummaryWriter(log_dir='./logs/%s/%s' % (args.experiment_name, x)) for x in ['train', 'test']]
    result = OrderedDict()
    epoch_start = 1
    if save_path and os.path.exists(save_path):
        logger.info('%s file found. loading...' % save_path)
        data = load_model(save_path, model, optimizer)
        if 'model' in data or 'state_dict' in data:
            logger.info('checkpoint epoch@%d' % data['epoch'])
            if data['epoch'] < config['epoch']:
                epoch_start = data['epoch']
            else:
                only_eval = True
        del data

    else:
        logger.info('"%s" file not found. skip to pretrain weights...' % save_path)
        if only_eval:
            logger.warning('model checkpoint not found. only-evaluation mode is off.')
        only_eval = False

    trainloader, unsuploader, testloader_ = get_dataloaders(args.ssl, args.config, args.dataroot, only_eval)
    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        rs = dict()
        _ = evaluate(result, rs, metric, 0, model, optimizer, testloader_,
                     criterion, config, 0, writers, scaler, save_path=None,
                     trainloader=trainloader)
        return result

    best_top1 = 0
    for epoch in range(epoch_start, config['epoch'] + 1):
        model.train()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, unsuploader, criterion, optimizer, config,
                                desc_default='train', epoch=epoch, writer=writers[0],
                                unsupervised=args.ssl, scaler=scaler, scheduler=scheduler)
        model.eval()
        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if epoch % verbose_eval == 0 or epoch == config['epoch']:
            best_top1 = evaluate(result, rs, metric, best_top1, model, optimizer,
                                 testloader_, criterion, config, epoch, writers,
                                 scaler, save_path, None)
    del model
    result['top1_test'] = best_top1
    return result


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    args = parser.parse_args()
    set_seed(args.seed)
    with open(args.config, 'r') as fp:
        args.config = yaml.load(fp)
    args.experiment_name = args.experiment_name or f'experiment_at_{time.time()}'
    if not args.only_eval:
        if args.save:
            logger.info('checkpoint will be saved at %s' % args.save)
        else:
            logger.warning('Provide --save argument to save the checkpoint. Without it, '
                           'training result will not be saved!')
    if args.save:
        add_filehandler(logger, os.path.join('./', f'{args.experiment_name}_log'))

    logger.info(json.dumps(args.config, indent=4))
    t = time.time()
    results = train_and_eval(args)
    elapsed = time.time() - t

    logger.info('done.')
    logger.info('model: %s' % args.config['model'])
    logger.info('augmentation: %s' % args.config.get('aug', None))
    logger.info('\n' + json.dumps(results, indent=4))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - results['top1_test']))
