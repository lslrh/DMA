import os
import torch
import imageio
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from fusion_util import extract_openseg_img_feature, PointCloudToImageMapper, save_fused_feature, get_matterport_camera_data
import torch.nn.functional as F
import open3d as o3d
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on Matterport3D.')
    parser.add_argument('--data_dir', type=str, default='/home/liruihuang/openscene/data_matterport', help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='train', help='split: "train"| "val" | "test" ')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--img_feat_dir', type=str, default='', help='the id range to process')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(data_path, out_dir, args):
    '''Process one scene.'''

    # short hand
    num_rand_file_per_scene = args.num_rand_file_per_scene
    feat_dim = args.feat_dim
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale


    # load 3D data (point cloud, color and the corresponding labels)
    locs_in = torch.load(data_path)[0]
    labels_in = torch.load(data_path)[2]
    n_points = locs_in.shape[0]

    # obtain all camera views related information (specificially for Matterport)
    intrinsics, poses, img_dirs, scene_id, num_img = \
            get_matterport_camera_data(data_path, locs_in, args)
    if num_img == 0:
        print('no views inside {}'.format(scene_id))
        return 1

    device = 'cuda'

    n_points_cur = n_points

    n_classes = 160
    pred_cls_num = torch.zeros((n_points_cur, n_classes), device=device)

    ################ Feature Fusion ###################
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    for img_id, img_dir in enumerate(tqdm(img_dirs)):
        # load pose
        pose = poses[img_id]

        # load per-image intrinsic
        intr = intrinsics[img_id]

        # load depth and convert to meter
        depth_dir = img_dir.replace('color', 'depth')
        _, img_type, yaw_id = img_dir.split('/')[-1].split('_')
        depth_dir = depth_dir[:-8] + 'd'+img_type[1] + '_' + yaw_id[0] + '.png'
        depth = imageio.v2.imread(depth_dir) / depth_scale

        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth, intr)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue
        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        vis_id[:, img_id] = mask

        semantic_mask = torch.from_numpy(np.load(img_dir.replace('matterport_2d', 'matterport_semantic_label_fcclip').replace('color/','').replace('jpg', 'npy'))).to(device)

        label_one_hot = F.one_hot(semantic_mask.long(), n_classes)
        # img = imageio.v2.imread(img_dir)
        # visualize_2d(img, semantic_mask.numpy(), semantic_mask.shape, './semantic_mask.png')
        # visualize_partition_2d(semantic_sam_mask)
        # visualize_2d(img, semantic_sam_mask*(semantic_sam_mask==6), semantic_mask.shape, './semantic_sam_mask.png')
        label_2d_3d = label_one_hot[mapping[:, 1], mapping[:, 2], :] 
        pred_cls_num[mask!=0] += label_2d_3d[mask!=0]

    value, label_3d = torch.max(pred_cls_num, dim=-1)
    label_3d[value==0] = 255
    label_3d = label_3d.cpu().numpy()

    save_path = data_path.replace('matterport_3d', 'matterport_3d_fcclip_paint')
    torch.save(label_3d, save_path)
    # labels_in[value.cpu().numpy()==0] = 255
    visualize_partition(locs_in, label_3d, save_path.replace('.pth', '.pc.ply'))


def visualize_partition(coord, group_id, save_path):
    SCANNET_COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                    6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                    13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}

    colors = np.array(list(SCANNET_COLOR_MAP_20.values()))[1:]

    num_groups = group_id.max() + 1
    group_colors = np.random.rand(num_groups, 3)
    segmentation_color = np.zeros((len(coord), 3))
    for i, color in enumerate(group_colors):
        segmentation_color[group_id == i] = color
    save_point_cloud(coord, segmentation_color, save_path)

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
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(coord) if color is None else color)
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")    




def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #### Dataset specific parameters #####
    img_dim = (640, 512)
    depth_scale = 4000.0
    #######################################
    visibility_threshold = 0.02 # threshold for the visibility check

    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    args.keep_features_in_memory = False # keep image features in the memory, very expensive
    args.feat_dim = 768 # CLIP feature dimension
    split = args.split
    data_dir = args.data_dir

    data_root = join(data_dir, 'matterport_3d')
    data_root_2d = join(data_dir,'matterport_2d')
    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range


    if split== 'train': # for training set, export a chunk of point cloud
        args.n_split_points = 20000
        args.num_rand_file_per_scene = 5
    else: # for the validation set, export the entire point cloud instead of chunks
        args.n_split_points = 2000000
        args.num_rand_file_per_scene = 1

    # calculate image pixel-3D points correspondances
    args.point2img_mapper = PointCloudToImageMapper(
            image_dim=img_dim,
            visibility_threshold=visibility_threshold,
            cut_bound=args.cut_num_pixel_boundary)

    data_paths = sorted(glob(join(data_root, split, '*.pth')))
    total_num = len(data_paths)

    id_range = None
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]

    for i in trange(total_num):
        if id_range is not None and \
           (i<id_range[0] or i>id_range[1]):
            print('skip ', i, data_paths[i])
            continue

        process_one_scene(data_paths[i], out_dir, args)

if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)

    main(args)
## python matterport_openseg.py --data_dir ../../data --output_dir ../../data/matterport_multiview_openseg --openseg_model ~/workspace/openseg_exported_clip --split train