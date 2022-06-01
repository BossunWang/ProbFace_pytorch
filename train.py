"""
@author: Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""
import os
import argparse
import logging as logger
import random
import numpy as np
from contextlib import contextmanager
import sys
sys.path.append("face_mask_adding/FMA-3D")

import torch
import torch.distributed as dist
import torch.utils.data.distributed
from torch import optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import set_start_method
from tensorboardX import SummaryWriter

from data.train_dataset import ImageDataset
from model import backbone, uncertainty_head
from loss import MLSloss, triplet_semihard_loss
from utils.AverageMeter import AverageMeter
from models.prnet import PRNet


logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

adaface_models = {
    'ir_50': "pretrained/adaface_ir50_webface4m.ckpt",
    'ir_101': "pretrained/adaface_ir101_webface12m.ckpt",
}


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    logger.info(s)  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def decay_lr(optimizer, scale=0.1):
    """Get the current learning rate from optimizer.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= scale


def get_lr(optimizer):
    """Get the current learning rate from optimizer. 
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def load_backbone_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = backbone.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    return model


def train_one_epoch(data_loader
                    , model, head
                    , optimizer, MLS, TripletSemiHard
                    , cur_epoch
                    , loss_meter, MLS_loss_meter, triplet_semihard_loss_meter, output_constraint_loss_meter
                    , args, device):
    """Tain one epoch by traditional training.
    """

    for batch_idx, (images, labels) in enumerate(data_loader):
        global_batch_idx = cur_epoch * len(data_loader) + batch_idx
        images = images.to(device)
        labels = labels.to(device)
        batch_size, sample_size, ch, h, w = images.size()
        batch_sample_size = batch_size * sample_size
        shuffle_ids = torch.randperm(batch_size).long().to(device)
        images = images.view(-1, ch, h, w)[shuffle_ids]
        labels = labels.view(-1)[shuffle_ids]

        with torch.no_grad():
            features, feature_fusions = model(images)
            norm = torch.norm(features, 2, 1, True)
            features = torch.div(features, norm)

        log_sigma_sq = head(feature_fusions)

        MLS_loss, attention_mat, mean_pos, mean_neg = MLS(features.detach(), log_sigma_sq, labels)
        triplet_loss = TripletSemiHard(attention_mat, labels, margin=args.triplet_margin)
        sigma_sq = torch.exp(log_sigma_sq)
        sigma_sq_m = sigma_sq.mean().detach()
        output_constraint_loss = ((sigma_sq / sigma_sq_m - 1.) ** 2.).mean()

        sigma_sq_max = torch.max(sigma_sq)
        sigma_sq_min = torch.min(sigma_sq)

        loss = args.mls_loss_weight * MLS_loss \
               + args.discriminate_loss_weight * triplet_loss \
               + args.output_constraint_loss_weight * output_constraint_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_batch_idx in args.milestones:
            decay_lr(optimizer)

        loss_meter.update(loss.item(), batch_sample_size)
        MLS_loss_meter.update(MLS_loss.item(), batch_sample_size)
        triplet_semihard_loss_meter.update(triplet_loss.item(), batch_sample_size)
        output_constraint_loss_meter.update(output_constraint_loss, batch_sample_size)

        if args.local_rank in [-1, 0] and batch_idx % args.print_freq == 0:
            MLS_loss_avg = MLS_loss_meter.avg
            triplet_semihard_loss_avg = triplet_semihard_loss_meter.avg
            output_constraint_loss_avg = output_constraint_loss_meter.avg
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info('Epoch %d, iter %d/%d, lr %f, MLS loss %.4f, triplet loss %.4f, constraint_loss %.4f, loss %.4f'
                        ', mean_pos %.4f, mean_neg %.4f, sigma_sq_max %.4f, sigma_sq_min %.4f' %
                        (cur_epoch, batch_idx, len(data_loader), lr
                         , MLS_loss_avg, triplet_semihard_loss_avg, output_constraint_loss_avg, loss_avg
                         , mean_pos, mean_neg, sigma_sq_max, sigma_sq_min))
            args.writer.add_scalar('Train_loss', loss_avg, global_batch_idx)
            args.writer.add_scalar('Train_MLS_loss', MLS_loss_avg, global_batch_idx)
            args.writer.add_scalar('Train_triplet_semihard_loss', triplet_semihard_loss_avg, global_batch_idx)
            args.writer.add_scalar('Train_output_constraint_loss', output_constraint_loss_avg, global_batch_idx)
            args.writer.add_scalar('Train_mean_pos', mean_pos, global_batch_idx)
            args.writer.add_scalar('Train_mean_neg', mean_neg, global_batch_idx)
            args.writer.add_scalar('Train_sigma_sq_max', sigma_sq_max, global_batch_idx)
            args.writer.add_scalar('Train_sigma_sq_min', sigma_sq_min, global_batch_idx)
            args.writer.add_scalar('Train_lr', lr, global_batch_idx)

            MLS_loss_meter.reset()
            triplet_semihard_loss_meter.reset()
            output_constraint_loss_meter.reset()
            loss_meter.reset()

        if args.local_rank in [-1, 0] and (batch_idx + 1) % args.save_freq == 0:
            saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
            state = {
                'state_dict': head.module.state_dict(),
                'epoch': cur_epoch,
                'batch_id': batch_idx,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.out_dir, saved_name))
            logger.info('Save checkpoint %s to disk.' % saved_name)

    if args.local_rank in [-1, 0]:
        saved_name = 'Epoch_%d.pt' % cur_epoch
        state = {'state_dict': head.module.state_dict(),
                 'epoch': cur_epoch,
                 'batch_id': batch_idx,
                 'optimizer': optimizer.state_dict()}
        torch.save(state, os.path.join(args.out_dir, saved_name))
        logger.info('Save checkpoint %s to disk...' % saved_name)


def train(args):
    """Total training procedure.
    """
    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(log_dir=args.tensorboardx_logdir)
        args.writer = writer
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    device = select_device(args.device, batch_size=args.batch_size)
    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert args.batch_size % args.world_size == 0, '--batch-size must be multiple of CUDA device count'
        args.batch_size = args.batch_size // args.world_size

    cuda = device.type != 'cpu'
    init_seeds(2 + args.local_rank)

    model = load_backbone_pretrained_model(args.backbone_type)
    model = model.to(device)
    model.eval()

    prnet = PRNet(3, 3).to(device)
    prnet_model_path = "face_mask_adding/FMA-3D/models/prnet.pth"
    state_dict = torch.load(prnet_model_path)
    prnet.load_state_dict(state_dict)
    prnet.eval()

    for param in model.parameters():
        param.requires_grad = False

    in_feat = 47040
    unh = uncertainty_head.UncertaintyHead(in_feat=in_feat).to(device)
    unh.train()

    if args.resume:
        ckpt = torch.load(args.pretrain_model, map_location=device)  # load checkpoint
        state_dict = ckpt['state_dict']
        unh.load_state_dict(state_dict, strict=True)  # load

    # Optimizer
    MLS = MLSloss.MLSLoss()
    TripletSemiHard = triplet_semihard_loss.TripletSemiHardLoss()
    ori_epoch = 0
    optimizer = optim.SGD(unh.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    loss_meter = AverageMeter()
    MLS_loss_meter = AverageMeter()
    triplet_semihard_loss_meter = AverageMeter()
    output_constraint_loss_meter = AverageMeter()

    if args.resume:
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        ori_epoch = ckpt['epoch']

    # DP mode
    if cuda and args.local_rank == -1 and torch.cuda.device_count() > 1:
        unh = torch.nn.DataParallel(unh)

    # SyncBatchNorm
    if args.sync_bn and cuda and args.local_rank != -1:
        unh = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unh).to(device)
        logger.info('Using SyncBatchNorm()')

    # DDP mode
    if cuda and args.local_rank != -1:
        unh = DDP(unh, device_ids=[args.local_rank], output_device=args.local_rank)

    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(args.local_rank):
        trainset = ImageDataset(args.data_root
                                , args.train_file
                                , sample_size=args.sample_size
                                , data_aug=args.data_aug
                                , masked_ratio=args.masked_ratio
                                , device=device
                                , prnet_model=prnet)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset) if args.local_rank != -1 else None
    train_loader = DataLoader(dataset=trainset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True)

    for epoch in range(ori_epoch, args.epoches):
        train_one_epoch(train_loader
                        , model, unh
                        , optimizer, MLS, TripletSemiHard
                        , epoch
                        , loss_meter, MLS_loss_meter, triplet_semihard_loss_meter, output_constraint_loss_meter
                        , args, device)

    if args.local_rank != -1:
        dist.destroy_process_group()


if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='traditional_training for face recognition.')
    conf.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    conf.add_argument("--data_root", type=str,
                      help="The root folder of training set.")
    conf.add_argument("--train_file", type=str,
                      help="The training file path.")
    conf.add_argument("--backbone_type", type=str,
                      help="Mobilefacenets, Resnet.")
    conf.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    conf.add_argument('--lr', type=float, default=0.1,
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type=str,
                      help="The folder to save models.")
    conf.add_argument('--epoches', type=int, default=9,
                      help='The training epoches.')
    conf.add_argument('--step', type=str, default='2,5,7',
                      help='Step for lr.')
    conf.add_argument('--workers', type=int, default=4)
    conf.add_argument('--print_freq', type=int, default=10,
                      help='The print frequency for training state.')
    conf.add_argument('--save_freq', type=int, default=10,
                      help='The save frequency for training state.')
    conf.add_argument('--batch_size', type=int, default=8,
                      help='The training batch size over all gpus.')
    conf.add_argument('--sample_size', type=int, default=16,
                      help='The training batch size over all gpus.')
    conf.add_argument('--data_aug', action='store_true')
    conf.add_argument('--momentum', type=float, default=0.9,
                      help='The momentum for sgd.')
    conf.add_argument('--triplet_margin', type=float, default=3.0)
    conf.add_argument('--masked_ratio', type=float, default=0.5)
    conf.add_argument('--mls_loss_weight', type=float, default=1.0)
    conf.add_argument('--output_constraint_loss_weight', type=float, default=0.001)
    conf.add_argument('--discriminate_loss_weight', type=float, default=0.001)
    conf.add_argument('--log_dir', type=str, default='log',
                      help='The directory to save log.log')
    conf.add_argument('--tensorboardx_logdir', type=str,
                      help='The directory to save tensorboardx logs')
    conf.add_argument('--pretrain_model', type=str, default='',
                      help='The path of pretrained model')
    conf.add_argument('--resume', '-r', action='store_true', default=False,
                      help='Whether to resume from a checkpoint.')
    conf.add_argument('--sync_bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    args = conf.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    train(args)
    logger.info("training done")
