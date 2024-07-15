import os
import time
import random
import numpy as np
import logging
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from MinkowskiEngine import SparseTensor
from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, \
    poly_learning_rate, poly_weight, save_checkpoint, \
    export_pointcloud, get_palette, convert_labels_with_palette, extract_clip_feature
from dataset.label_constants import *
from dataset.feature_loader import FusedFeatureLoader, collation_fn
from dataset.point_loader import Point3DLoader, collation_fn_eval_all
from models.disnet import DisNet as Model
from tqdm import tqdm
import torch_scatter
import copy
import torch.nn.functional as F

best_iou = 0.0


def worker_init_fn(worker_id):
    '''Worker initialization.'''
    random.seed(time.time() + worker_id)


def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='OpenScene 3D distillation.')
    parser.add_argument('--config', type=str,
                        default='config/scannet/distill_openseg.yaml',
                        help='config file')
    parser.add_argument('opts',
                        default=None,
                        help='see config/scannet/distill_openseg.yaml for all options',
                        nargs=argparse.REMAINDER)
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    model_dir = os.path.join(cfg.save_path, 'model')
    result_dir = os.path.join(cfg.save_path, 'result')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + '/last', exist_ok=True)
    os.makedirs(result_dir + '/best', exist_ok=True)
    return cfg


def get_logger():
    '''Define logger.'''

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    '''Main function.'''

    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    
    # By default we use shared memory for training
    if not hasattr(args, 'use_shm'):
        args.use_shm = True

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node,
                 args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    global best_iou
    args = argss

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)

    model = get_model(args)
    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")

    # ####################### Optimizer ####################### #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    args.index_split = 0

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module.'):
                    # remove module
                    k = k[7:]
                # else:
                #     # add module
                #     k = 'module.' + k

                new_state_dict[k]=v
            model.load_state_dict(new_state_dict, strict=True)
            # model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info(
                    "=> no checkpoint found at '{}'".format(args.resume))

    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'):
        # by default we do not use the point color as input
        args.input_color = False
    train_data = FusedFeatureLoader(datapath_prefix=args.data_root,
                                    datapath_prefix_feat=args.data_root_2d_fused_feature,
                                    voxel_size=args.voxel_size,
                                    split='train', aug=args.aug,
                                    memcache_init=args.use_shm, loop=args.loop,
                                    input_color=args.input_color
                                    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                            shuffle=(train_sampler is None),
                                            num_workers=args.workers, pin_memory=True,
                                            sampler=train_sampler,
                                            drop_last=True, collate_fn=collation_fn,
                                            worker_init_fn=worker_init_fn)
    if args.evaluate:
        val_data = Point3DLoader(datapath_prefix=args.data_root,
                                 voxel_size=args.voxel_size,
                                 split='val', aug=False,
                                 memcache_init=args.use_shm,
                                 eval_all=True,
                                 input_color=args.input_color)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data) if args.distributed else None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                shuffle=False,
                                                num_workers=args.workers, pin_memory=True,
                                                drop_last=False, collate_fn=collation_fn_eval_all,
                                                sampler=val_sampler)

        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu) # for evaluation

    # ####################### Distill ####################### #
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.evaluate:
                val_sampler.set_epoch(epoch)

        # loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
        #         val_loader, model, criterion)
        # import pdb; pdb.set_trace()        
        loss_train = distill(train_loader, model, optimizer, epoch)
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
                val_loader, model, criterion)
            # raise NotImplementedError

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                # remember best iou and save checkpoint
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            save_checkpoint(
                {
                    'epoch': epoch_log,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_iou': best_iou
                }, is_best, os.path.join(args.save_path, 'model')
            )

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def get_model(cfg):
    '''Get the 3D model.'''

    model = Model(cfg=cfg)
    return model


def obtain_text_features_and_palette():
    '''obtain the CLIP text feature and palette.'''

    if 'scannet_200' in args.data_root:
        labelset = list(SCANNET_LABELS_200)
        palette = get_palette()
        dataset_name = 'scannet_200'
    elif 'scannet' in args.data_root:
        labelset = list(SCANNET_LABELS_20)
        # labelset = [ "a " + label + " in a scene" for label in labelset]
        labelset[-1] = 'other'
        palette = get_palette()
        dataset_name = 'scannet'
    elif 'matterport' in args.data_root:
        labelset = list(MATTERPORT_LABELS_21)
        palette = get_palette(colormap='matterport')
        dataset_name = 'matterport'
    elif 'nuscenes' in args.data_root:
        labelset = list(NUSCENES_LABELS_16)
        palette = get_palette(colormap='nuscenes16')
        dataset_name = 'nuscenes'

    if not os.path.exists('saved_text_embeddings'):
        os.makedirs('saved_text_embeddings')

    if 'openseg' in args.feature_2d_extractor:
        model_name="ViT-L/14@336px"
        postfix = '_768' # the dimension of CLIP features is 768
    elif 'lseg' in args.feature_2d_extractor:
        model_name="ViT-B/32"
        postfix = '_512' # the dimension of CLIP features is 512
    else:
        raise NotImplementedError

    # clip_file_name = 'saved_text_embeddings/clip_{}_labels{}.pt'.format(dataset_name, postfix)
    clip_file_name = 'saved_text_embeddings/clip_{}_labels{}.pt'.format(dataset_name, postfix)
    # clip_file_name = 'saved_text_embeddings/clip_scannet_fcclip_labels_768_2.pt'
    try: # try to load the pre-saved embedding first
    # logger.info('Load pre-computed embeddings from {}'.format(clip_file_name))
        text_features = torch.load(clip_file_name, map_location='cuda').cuda()
    except: # extract CLIP text features and save them
        text_features = extract_clip_feature(labelset, model_name=model_name)
        torch.save(text_features, clip_file_name)
    
    # text_features = torch.load(clip_file_name)[:-1].cuda().half()
    # num_templates = [1, 1, 3, 3, 1, 1, 1, 1, 2, 3, 4, 1, 2, 3, 1, 3, 4, 3, 8, 6]
    # text_features_list = []
    # cur_idx = 0
    # for num_t in num_templates:   
    #     text_features_list.append(text_features[cur_idx:cur_idx + num_t, :].mean(0))
    #     cur_idx += num_t
    # text_features = torch.stack(text_features_list, dim=0)
    return text_features, palette

def num_to_natural(group_ids):
    '''
    Change the group number to natural number arrangement
    '''
    group_ids = group_ids.numpy()
    if np.all(group_ids == -1):
        return torch.from_numpy(group_ids)
    array = copy.deepcopy(group_ids)
    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return torch.from_numpy(array)

def contrastive(prototypes, feats, labels):
    mask = torch.zeros(len(labels), len(prototypes)).scatter_(1, labels.cpu(), 1).half().cuda(non_blocking=True)
    feats = F.normalize(feats, dim=1, p=2, eps=1e-5).half()
    proto = F.normalize(prototypes, dim=1, p=2, eps=1e-5)
    target_ema = torch.mm(feats, proto.half().t())
    
    anchor_dot_contrast = target_ema/0.1
    
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

    logits = anchor_dot_contrast - logits_max.detach()

    neg_mask = 1 - mask
    neg_logits = torch.exp(logits) * neg_mask
    neg_logits = neg_logits.sum(1, keepdim=True)

    exp_logits = torch.exp(logits)

    log_prob = logits - torch.log(exp_logits + neg_logits)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1e-5)
    loss = - mean_log_prob_pos
    loss = loss.mean()
    return loss

def scatter_topk(idx, k=1):
    mask = torch.ones_like(idx, dtype=torch.bool)
    mask[k:] = idx[k:] != idx[:-k]
    return mask


def distill(train_loader, model, optimizer, epoch):
    '''Distillation pipeline.'''

    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    text_features, palette = obtain_text_features_and_palette()

    # start the distillation process
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        (coords, feat, label_3d, painted_label_3d, feat_3d, mask) = batch_data
        
        # change the painted_label_id
        # unique_index, inverse_indices = torch.unique(coords[:,0], return_inverse=True)
        # max_id = [torch.max(painted_label_3d[inverse_indices == idx]).item() \
        #           for idx in range(len(unique_index)) if torch.sum(inverse_indices == idx) > 0]
        # max_id[0:0] = [0]
        # max_id.pop()
        # sum_id = torch.cumsum(torch.tensor(max_id), dim=0)+1
        # sum_id[0] = 0
        # painted_label_3d_new = painted_label_3d + sum_id[coords[:,0].long()]
        # painted_label_3d_new[painted_label_3d==-1] = -1
        # painted_label_3d = painted_label_3d_new

        coords[:, 1:4] += (torch.rand(3) * 100).type_as(coords)
        sinput = SparseTensor(
            feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
        feat_3d, mask = feat_3d.cuda(
            non_blocking=True), mask.cuda(non_blocking=True)
        
        output_3d = model(sinput)
        output_3d = output_3d[mask]
        painted_label_3d = painted_label_3d[mask].cuda()

        mask_valid = painted_label_3d != -1
        painted_label_3d = painted_label_3d[mask_valid]
        # painted_label_3d = num_to_natural(painted_label_3d)
        output_3d_all = output_3d[mask_valid]
        feat_3d_all = feat_3d[mask_valid]

        # feat_similarity = F.softmax(torch.mm(F.normalize(feat_3d_all, 2, 1, eps=1e-5), text_features.detach().T)/0.001, dim=1)
        # pred_mask = torch.argmax(feat_similarity, dim=1)
        # weight = (pred_mask == painted_label_3d).detach()

        # pred_kl_loss = F.cross_entropy(out_similarity.float(), feat_similarity.float())
        # pred_kl_loss = F.cross_entropy(out_similarity[weight].float(), painted_label_3d[weight]) 
        
        pred_kl_loss = (1 - torch.nn.CosineSimilarity()(F.normalize(output_3d_all, 2, 1, eps=1e-5), text_features[painted_label_3d])).mean()

        if hasattr(args, 'loss_type') and args.loss_type == 'cosine':
            # loss = (1 - torch.nn.CosineSimilarity()(output_3d_all, feat_3d_all)).mean() #+ 1 * pred_kl_loss
            loss = pred_kl_loss
        elif hasattr(args, 'loss_type') and args.loss_type == 'l1':
            loss = torch.nn.L1Loss()(output_3d, feat_3d)
            # loss += torch.nn.L1Loss()(feat_3d_center_mask, feat_3d_center_remain)
        else:
            raise NotImplementedError

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)

        # adjust learning rate
        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(
            args.base_lr, current_iter, max_iter, power=args.power)

        for index in range(0, args.index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(
            int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                            batch_time=batch_time, data_time=data_time,
                                                            remain_time=remain_time,
                                                            loss_meter=loss_meter))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)

        end = time.time()

        # mask_first = (coords[mask][:, 0] == 0)
        # # output_3d = output_3d[mask_first]
        # feat_3d = feat_3d[mask_first]
        # # logits_pred = output_3d.half() @ text_features.t()
        # logits_img = feat_3d.half() @ text_features.t()
        # # logits_pred = torch.max(logits_pred, 1)[1].cpu().numpy()

        # pcl = coords[:, 1:].cpu().numpy()
        # num_templates = [1, 1, 3, 3, 1, 1, 1, 1, 2, 3, 4, 1, 2, 3, 1, 3, 4, 3, 8, 6]
        # final_pred_logits = []
        # cur_idx = 0
        # for num_t in num_templates:     
        #     final_pred_logits.append(logits_img[:, cur_idx: cur_idx + num_t].max(-1).values)
        #     cur_idx += num_t
        # logits_img = torch.stack(final_pred_logits, dim=-1)

        # logits_img = torch.max(logits_img, 1)[1].cpu().numpy()
        # mask = mask.cpu().numpy()
        # logits_gt = label_3d.numpy()[mask][mask_first.cpu().numpy()]
        # logits_gt[logits_gt == 255] = args.classes


        # seg_label_color = convert_labels_with_palette(logits_img, palette)
        # # pred_label_color = convert_labels_with_palette(
        # #     logits_pred, palette)
        # gt_label_color = convert_labels_with_palette(
        #     logits_gt, palette)
        # pcl_part = pcl[mask][mask_first.cpu().numpy()]

        # export_pointcloud(os.path.join(args.save_path, 'result', 'last', '{}_{}.ply'.format(args.feature_2d_extractor, epoch)), pcl_part, colors=seg_label_color)
        # # export_pointcloud(os.path.join(args.save_path, 'result', 'last',
        # #                     'pred_{}.ply'.format(epoch)), pcl_part, colors=pred_label_color)
        # export_pointcloud(os.path.join(args.save_path, 'result', 'last',
        #                     'gt_{}.ply'.format(epoch)), pcl_part, colors=gt_label_color)
        # import pdb; pdb.set_trace()

    return loss_meter.avg


def validate(val_loader, model, criterion):
    '''Validation.'''

    torch.backends.cudnn.enabled = False
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # obtain the CLIP feature
    # clip_file_name = 'saved_text_embeddings/clip_matterport_labels_768_21.pt'
    # text_features = torch.load(clip_file_name, map_location='cuda').cuda()

    text_features, _ = obtain_text_features_and_palette()
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            (coords, feat, label, inds_reverse) = batch_data
            sinput = SparseTensor(
                feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            label = label.cuda(non_blocking=True)
            output = model(sinput)
            output = output[inds_reverse, :]
            output = output.half() @ text_features.t()

            # num_templates = [1, 1, 3, 3, 1, 1, 1, 1, 2, 3, 4, 1, 2, 3, 1, 3, 4, 3, 8, 6]
            # final_pred_logits = []
            # cur_idx = 0
            # for num_t in num_templates:     
            #     final_pred_logits.append(output[:, cur_idx: cur_idx + num_t].max(-1).values)
            #     cur_idx += num_t
            # output = torch.stack(final_pred_logits, dim=-1)
            if 'scannet_200' in args.data_root:
                label[label==-1] = 255

            loss = criterion(output, label)
            output = torch.max(output, 1)[1]

            intersection, union, target = intersectionAndUnionGPU(output, label.detach(),
                                                                  args.classes, args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(
                    union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu(
            ).numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(
                union), target_meter.update(target)
            loss_meter.update(loss.item(), args.batch_size)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()
