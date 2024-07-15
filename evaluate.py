import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import random
import numpy as np
import logging
import argparse
import urllib

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from util import metric
from torch.utils import model_zoo

from MinkowskiEngine import SparseTensor
from util import config
from util.util import export_pointcloud, get_palette, \
    convert_labels_with_palette, extract_text_feature, visualize_labels
from tqdm import tqdm
from distill import get_model

from dataset.label_constants import *
import open3d as o3d

def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='OpenScene evaluation')
    parser.add_argument('--config', type=str,
                    default='config/scannet/eval_openseg.yaml',
                    help='config file')
    parser.add_argument('opts',
                    default=None,
                    help='see config/scannet/test_ours_openseg.yaml for all options',
                    nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
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

def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')

def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def precompute_text_related_properties(labelset_name):
    '''pre-compute text features, labelset, palette, and mapper.'''
    if 'head' in labelset_name:
        labelset = list(HEAD_CATS_SCANNET_200)
        palette = get_palette(colormap='scannet_200')
    elif 'common' in labelset_name:
        labelset = list(COMMON_CATS_SCANNET_200)
        palette = get_palette(colormap='scannet_200')
    elif 'tail' in labelset_name:
        labelset = list(TAIL_CATS_SCANNET_200)
        palette = get_palette(colormap='scannet_200')
    elif 'scannet_200' in labelset_name:
        labelset = list(SCANNET_LABELS_200)
        palette = get_palette(colormap='scannet_200')
    elif 'scannet' in labelset_name:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = 'other' # change 'other furniture' to 'other'
        palette = get_palette(colormap='scannet')
    elif labelset_name == 'matterport_3d' or labelset_name == 'matterport':
        labelset = list(MATTERPORT_LABELS_21)
        palette = get_palette(colormap='matterport')
    elif 'matterport_3d_40' in labelset_name or labelset_name == 'matterport40':
        labelset = list(MATTERPORT_LABELS_40)
        palette = get_palette(colormap='matterport_160')
    elif 'matterport_3d_80' in labelset_name or labelset_name == 'matterport80':
        labelset = list(MATTERPORT_LABELS_80)
        palette = get_palette(colormap='matterport_160')
    elif 'matterport_3d_160' in labelset_name or labelset_name == 'matterport160':
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')
    elif 'nuscenes' in labelset_name:
        labelset = list(NUSCENES_LABELS_16)
        palette = get_palette(colormap='nuscenes16')
    else: # an arbitrary dataset, just use a large labelset
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')

    mapper = None
    # if hasattr(args, 'map_nuscenes_details'):
    #     labelset = list(NUSCENES_LABELS_DETAILS)
    #     mapper = torch.tensor(MAPPING_NUSCENES_DETAILS, dtype=int)

    text_features = extract_text_feature(labelset, args)
    # labelset.append('unknown')
    labelset.append('unlabeled')
    return text_features, labelset, mapper, palette

def main():
    '''Main function.'''

    args = get_parser()

    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed    
    args.ngpus_per_node = len(args.test_gpu)
    if len(args.test_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    
    # By default we do not use shared memory for evaluation
    if not hasattr(args, 'use_shm'):
        args.use_shm = False
    if args.use_shm:
        if args.multiprocessing_distributed:
            args.world_size = args.ngpus_per_node * args.world_size
            mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.test_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    model = get_model(args)
    if main_process():
        global logger
        logger = get_logger()
        logger.info(args)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.test_workers = int(args.test_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    if args.feature_type == 'fusion':
        pass # do not need to load weight
    elif is_url(args.model_path): # load from url
        checkpoint = model_zoo.load_url(args.model_path, progress=True)
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        except Exception as ex:
            # The model was trained in a parallel manner, so need to be loaded differently
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module.'):
                    # remove module
                    k = k[7:]
                else:
                    # add module
                    k = 'module.' + k

                new_state_dict[k]=v
            model.load_state_dict(new_state_dict, strict=True)
            logger.info('Loaded a parallel model')
    
    elif args.model_path is not None and os.path.isfile(args.model_path):
        # load from directory
        if main_process():
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        except Exception as ex:
            # The model was trained in a parallel manner, so need to be loaded differently
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module.'):
                    # remove module
                    k = k[7:]
                else:
                    # add module
                    k = 'module.' + k

                new_state_dict[k]=v
            model.load_state_dict(new_state_dict, strict=True)
            logger.info('Loaded a parallel model')

        if main_process():
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))    
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'):
        # by default we do not use the point color as input
        args.input_color = False
    
    from dataset.feature_loader import FusedFeatureLoader, collation_fn_eval_all
    val_data = FusedFeatureLoader(datapath_prefix=args.data_root,
                                datapath_prefix_feat=args.data_root_2d_fused_feature,
                                voxel_size=args.voxel_size, 
                                split=args.split, aug=False,
                                memcache_init=args.use_shm, eval_all=True, identifier=6797,
                                input_color=args.input_color)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size,
                                                shuffle=False, num_workers=args.test_workers, pin_memory=True,
                                                drop_last=False, collate_fn=collation_fn_eval_all,
                                                sampler=val_sampler)

    # ####################### Test ####################### #
    labelset_name = args.data_root.split('/')[-1]
    if hasattr(args, 'labelset'):
        # if the labelset is specified
        labelset_name = args.labelset
    evaluate(model, val_loader, labelset_name, args.class_split)

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x

def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    if color is not None:
        color = to_numpy(color)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(color)
    # pcd.colors = o3d.utility.Vector3dVector(np.ones_like(coord) if color is None else color)
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")  

def evaluate(model, val_data_loader, labelset_name='scannet_3d', class_split=None):
    '''Evaluate our OpenScene model.'''

    torch.backends.cudnn.enabled = False

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)

    if args.save_feature_as_numpy: # save point features to folder
        out_root = os.path.commonprefix([args.save_folder, args.model_path])
        saved_feature_folder = os.path.join(out_root, 'saved_feature')
        os.makedirs(saved_feature_folder, exist_ok=True)

    # short hands
    save_folder = args.save_folder
    feature_type = args.feature_type
    eval_iou = True
    if hasattr(args, 'eval_iou'):
        eval_iou = args.eval_iou
    mark_no_feature_to_unknown = False
    if hasattr(args, 'mark_no_feature_to_unknown') and args.mark_no_feature_to_unknown and feature_type == 'fusion':
        # some points do not have 2D features from 2D feature fusion. Directly assign 'unknown' label to those points during inference
        mark_no_feature_to_unknown = True
    vis_input = False
    if hasattr(args, 'vis_input') and args.vis_input:
        vis_input = True
    vis_pred = False
    if hasattr(args, 'vis_pred') and args.vis_pred:
        vis_pred = True
    vis_gt = False
    if hasattr(args, 'vis_gt') and args.vis_gt:
        vis_gt = True
    if not class_split:
        text_features, labelset, mapper, palette = \
            precompute_text_related_properties(labelset_name)
    else:
        text_features, labelset, mapper, palette = \
            precompute_text_related_properties(labelset_name+class_split)

    with torch.no_grad():
        model.eval()
        store = 0.0
        for rep_i in range(args.test_repeats):
            preds, gts = [], []
            val_data_loader.dataset.offset = rep_i
            if main_process():
                logger.info(
                    "\nEvaluation {} out of {} runs...\n".format(rep_i+1, args.test_repeats))

            # repeat the evaluation process
            # to account for the randomness in MinkowskiNet voxelization
            if rep_i>0:
                seed = np.random.randint(10000)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            if mark_no_feature_to_unknown:
                masks = []
            # clip_file_name = 'saved_text_embeddings/clip_nuscenes_labels_768.pt' 

            # text_features = torch.load(clip_file_name, map_location='cuda').cuda()
            mapper = None
            for i, (coords, feat, label, feat_3d, mask, inds_reverse) in enumerate(tqdm(val_data_loader)):
                sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
                coords = coords[inds_reverse, :]
                pcl = coords[:, 1:].cpu().numpy()
                
                # convert label to label_head
                if class_split == 'head':
                    label_head = torch.ones_like(label)*255
                    CLASS_LABELS = SCANNET_LABELS_200
                    CLASS_LABELS_HEAD = HEAD_CATS_SCANNET_200
                    N_CLASSES_HEAD = len(CLASS_LABELS_HEAD)
                    for i in range(N_CLASSES_HEAD):
                        label_name = CLASS_LABELS_HEAD[i]
                        index = CLASS_LABELS.index(label_name)
                        label_head[label==index] = i
                    label = label_head
                elif class_split == 'common':
                    label_common = torch.ones_like(label)*255
                    CLASS_LABELS = SCANNET_LABELS_200
                    CLASS_LABELS_COMMON = COMMON_CATS_SCANNET_200
                    N_CLASSES_COMMON = len(CLASS_LABELS_COMMON)
                    for i in range(N_CLASSES_COMMON):
                        label_name = CLASS_LABELS_COMMON[i]
                        index = CLASS_LABELS.index(label_name)
                        label_common[label==index] = i
                    label = label_common
                elif class_split == 'tail':
                    label_tail = torch.ones_like(label)*255
                    CLASS_LABELS = SCANNET_LABELS_200
                    CLASS_LABELS_TAIL = TAIL_CATS_SCANNET_200
                    N_CLASSES_TAIL = len(CLASS_LABELS_TAIL)
                    for i in range(N_CLASSES_TAIL):
                        label_name = CLASS_LABELS_TAIL[i]
                        index = CLASS_LABELS.index(label_name)
                        label_tail[label==index] = i
                    label = label_tail

                if feature_type == 'distill':
                    predictions = model(sinput)
                    predictions = predictions[inds_reverse, :]
                    pred = predictions.half() @ text_features.t()
                    logits_pred = torch.max(pred, 1)[1].cpu()
                    # paint_labels[paint_labels==-1]=0
                    # logits_pred = paint_labels
                elif feature_type == 'fusion':
                    from torch.nn import functional as F
                    feat_3d = F.normalize(feat_3d.float(), dim=-1)
                    # text_features = torch.load('saved_text_embeddings/clip_scannet_fcclip_labels_768_2.pt')[:-1]
                    predictions = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
                    pred = predictions.half() @ text_features.half().t()

                    # num_templates = [1, 1, 3, 3, 1, 1, 1, 1, 2, 3, 4, 1, 2, 3, 1, 3, 4, 3, 8, 6]
                    # final_pred_logits = []
                    # cur_idx = 0
                    # for num_t in num_templates:     
                    #     final_pred_logits.append(pred[:, cur_idx: cur_idx + num_t].max(-1).values)
                    #     cur_idx += num_t
                    # final_pred_logits = torch.stack(final_pred_logits, dim=-1)
                    # logits_pred = torch.max(final_pred_logits, 1)[1].detach().cpu()
                    logits_pred = torch.max(pred, 1)[1].detach().cpu()
                    if mark_no_feature_to_unknown:
                    # some points do not have 2D features from 2D feature fusion.
                    # Directly assign 'unknown' label to those points during inference.
                        # logits_pred[~mask[inds_reverse]] = len(labelset)-1
                        # logits_pred[~mask[inds_reverse]] = len(labelset)-1
                        label[~mask[inds_reverse]] = 255

                elif feature_type == 'ensemble':
                    feat_fuse = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
                    # pred_fusion = feat_fuse.half() @ text_features.t()
                    pred_fusion = (feat_fuse/(feat_fuse.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

                    predictions = model(sinput)
                    predictions = predictions[inds_reverse, :]
                    # pred_distill = predictions.half() @ text_features.t()
                    pred_distill = (predictions/(predictions.norm(dim=-1, keepdim=True)+1e-5)).half() @ text_features.t()

                    # logits_distill = torch.max(pred_distill, 1)[1].detach().cpu()
                    # mask_ensem = pred_distill<pred_fusion # confidence-based ensemble
                    # pred = pred_distill
                    # pred[mask_ensem] = pred_fusion[mask_ensem]
                    # logits_pred = torch.max(pred, 1)[1].detach().cpu()

                    feat_ensemble = predictions.clone().half()
                    mask_ =  pred_distill.max(dim=-1)[0] < pred_fusion.max(dim=-1)[0]
                    feat_ensemble[mask_] = feat_fuse[mask_]
                    pred = feat_ensemble @ text_features.t()
                    logits_pred = torch.max(pred, 1)[1].detach().cpu()

                    predictions = feat_ensemble # if we need to save the features
                else:
                    raise NotImplementedError

                if args.save_feature_as_numpy:
                    scene_name = val_data_loader.dataset.data_paths[i].split('/')[-1].split('.pth')[0]
                    np.save(os.path.join(saved_feature_folder, '{}_openscene_feat_{}.npy'.format(scene_name, feature_type)), predictions.cpu().numpy())
                
                # special case for nuScenes, evaluation points are only a subset of input
                if 'nuscenes' in labelset_name:
                    label_mask = (label!=255)
                    label = label[label_mask]
                    logits_pred = logits_pred[label_mask]
                    pred = pred[label_mask]
                    if vis_pred:
                        pcl = torch.load(val_data_loader.dataset.data_paths[i])[0][label_mask]

                # label_mask = (label!=255)
                # label = label[label_mask]
                # logits_pred = logits_pred[label_mask]
                # pred = pred[label_mask]

                if vis_input:
                    input_color = torch.load(val_data_loader.dataset.data_paths[i])[1]
                    export_pointcloud(os.path.join(save_folder, '{}_input.ply'.format(i)), pcl)
                    # export_pointcloud(os.path.join(save_folder, '{}_input.ply'.format(i)), pcl, colors=(input_color+1)/2)

                if vis_pred:
                    if mapper is not None:
                        pred_label_color = convert_labels_with_palette(mapper[logits_pred].numpy(), palette)
                        export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(i, feature_type)), pcl, colors=pred_label_color)
                    else:
                        pred_label_color = convert_labels_with_palette(logits_pred.numpy(), palette)
                        export_pointcloud(os.path.join(save_folder, '{}_{}.ply'.format(i, feature_type)), pcl, colors=pred_label_color)
                        visualize_labels(list(np.unique(logits_pred.numpy())),
                                    labelset,
                                    palette,
                                    os.path.join(save_folder, '{}_labels_{}.jpg'.format(i, feature_type)), ncol=5)

                # Visualize GT labels
                if vis_gt:
                    # for points not evaluating
                    label[label==255] = len(labelset)-1
                    gt_label_color = convert_labels_with_palette(label.cpu().numpy(), palette)
                    export_pointcloud(os.path.join(save_folder, '{}_gt.ply'.format(i)), pcl, colors=gt_label_color)
                    visualize_labels(list(np.unique(label.cpu().numpy())),
                                labelset,
                                palette,
                                os.path.join(save_folder, '{}_labels_gt.jpg'.format(i)), ncol=5)

                    # if 'nuscenes' in labelset_name:
                    #     all_digits = np.unique(np.concatenate([np.unique(mapper[logits_pred].numpy()), np.unique(label)]))
                    #     labelset = list(NUSCENES_LABELS_16)
                    #     labelset[4] = 'construct. vehicle'
                    #     labelset[10] = 'road'
                    #     visualize_labels(list(all_digits), labelset, 
                    #         palette, os.path.join(save_folder, '{}_label.jpg'.format(i)), ncol=all_digits.shape[0])

                if eval_iou:
                    if mark_no_feature_to_unknown:
                        if "nuscenes" in labelset_name: # special case
                            masks.append(mask[inds_reverse][label_mask])
                        else:
                            masks.append(mask[inds_reverse])

                    if args.test_repeats==1:
                        # save directly the logits
                        preds.append(logits_pred)
                    else:
                        # only save the dot-product results, for repeat prediction
                        preds.append(pred.cpu())
                    gts.append(label.cpu())

            if eval_iou:
                gt = torch.cat(gts)
                pred = torch.cat(preds)

                pred_logit = pred
                if args.test_repeats>1:
                    pred_logit = pred.float().max(1)[1]

                if mapper is not None:
                    pred_logit = mapper[pred_logit]

                if mark_no_feature_to_unknown:
                    mask = torch.cat(masks)
                    pred_logit[~mask] = 256

                if args.test_repeats==1:
                    current_iou = metric.evaluate(pred_logit.numpy(),
                                                gt.numpy(),
                                                dataset=labelset_name,
                                                stdout=True)
                if args.test_repeats > 1:
                    store = pred + store
                    store_logit = store.float().max(1)[1]
                    if mapper is not None:
                        store_logit = mapper[store_logit]

                    if mark_no_feature_to_unknown:
                        store_logit[~mask] = 256

                    if class_split:
                        accumu_iou = metric.evaluate_scannet200(store_logit.numpy(),
                                                gt.numpy(),
                                                stdout=True,
                                                dataset=labelset_name+class_split)
                    else:
                        accumu_iou = metric.evaluate(store_logit.numpy(),
                                                gt.numpy(),
                                                stdout=True,
                                                dataset=labelset_name)
                    

if __name__ == '__main__':
    main()
