import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import sys
sys.path.append("/home/liruihuang/openscene")
import torch
import imageio
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from os.path import join, exists
from fusion_util import extract_openseg_img_feature, PointCloudToImageMapper, save_fused_feature
import torch.nn.functional as F
from PIL import Image
import copy
import open3d as o3d

from util.util import extract_clip_feature

def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='train', help='split: "train"| "val"')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--openseg_model', type=str, default='', help='Where is the exported OpenSeg model')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--img_feat_dir', type=str, default='', help='the id range to process')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(data_path, args):
    '''Process one scene.'''
    # short hand
    scene_id = data_path.split('/')[-1].split('.')[0]

    point2img_mapper = args.point2img_mapper

    # load 3D data (point cloud)
    locs_in = torch.load(data_path)[0]
    labels_in = torch.load(data_path)[2].astype(int)
    n_points = locs_in.shape[0]

    cam_locs = ['back', 'back_left', 'back_right', 'front', 'front_left', 'front_right']

    # short hand for processing 2D features
    scene = join(args.data_root_2d, args.split, scene_id)
    img_dir_base = join(scene, 'color')
    pose_dir_base = join(scene, 'pose')
    K_dir_base = join(scene, 'K')
    num_img = len(cam_locs)
    device = 'cuda'

    n_points_cur = n_points
    n_classes = 16
    pred_cls_num = torch.zeros((n_points_cur, n_classes), device=device)

    ################ Feature Fusion ###################
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    for img_id, cam in enumerate(tqdm(cam_locs)):
        # load pose
        img_dir = join(img_dir_base, cam+'.jpg')
        intr = np.load(join(K_dir_base, cam+'.npy'))
        pose = np.load(join(pose_dir_base, cam+'.npy'))


        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth=None, intrinsic=intr)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        vis_id[:, img_id] = mask

        semantic_mask = torch.from_numpy(np.load(img_dir.replace(args.split, 'nuscenes_semantic_label_fcclip_v2/train').replace('color/','').replace('jpg', 'npy'))).to(device)
        label_one_hot = F.one_hot(semantic_mask.long(), n_classes)
        # img = imageio.v2.imread(img_dir)
        # visualize_2d(img, semantic_mask.numpy(), semantic_mask.shape, './semantic_mask.png')
        # visualize_partition_2d(semantic_sam_mask)
        # visualize_2d(img, semantic_sam_mask*(semantic_sam_mask==6), semantic_mask.shape, './semantic_sam_mask.png')
        # import pdb; pdb.set_trace()

        label_2d_3d = label_one_hot[mapping[:, 1], mapping[:, 2], :] 
        pred_cls_num[mask!=0] += label_2d_3d[mask!=0]
        
        # feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)

        # counter[mask!=0]+= 1
        # sum_features[mask!=0] += feat_2d_3d[mask!=0]
    value, label_3d = torch.max(pred_cls_num, dim=-1)
    label_3d[value==0] = 255
    label_3d = label_3d.cpu().numpy()

    labels_in[value.cpu().numpy()==0] = 255
    
    save_path = scene.replace(args.split, 'nuscenes_3d_fcclip_paint/'+args.split) + '.pth'
    # import pdb; pdb.set_trace()
    # print((label_3d[labels_in!=255]==labels_in[labels_in!=255]).sum()/len(labels_in[labels_in!=255]))
    # import pdb; pdb.set_trace()
    # visualize_partition(locs_in[labels_in!=255], labels_in[labels_in!=255]+1, save_path.replace('.pth', '.pc.ply'))
    # visualize_partition(locs_in[labels_in!=255], label_3d[labels_in!=255]+1, save_path.replace('.pth', '.pc.ply'))
    torch.save(label_3d, save_path)



def visualize_partition(coord, group_id, save_path):
    SCANNET_COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                    6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                    13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}
    # NUSCENES16_COLORMAP = {
    # 0: (0, 0, 0),
    # 1: (220,220,  0), # barrier
    # 2: (119, 11, 32), # bicycle
    # 3: (0, 60, 100), # bus
    # 4: (0, 0, 250), # car
    # 5: (230,230,250), # construction vehicle
    # 6: (0, 0, 230), # motorcycle
    # 7: (220, 20, 60), # person
    # 8: (250, 170, 30), # traffic cone
    # 9: (200, 150, 0), # trailer
    # 10: (0, 0, 110) , # truck
    # 11: (128, 64, 128), # road
    # 12: (0,250, 250), # other flat
    # 13: (244, 35, 232), # sidewalk
    # 14: (152, 251, 152), # terrain
    # 15: (70, 70, 70), # manmade
    # 16: (107,142, 35), # vegetation
    # }
    # colors = np.array(list(SCANNET_COLOR_MAP_20.values()))
    num_groups = group_id.max()
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

def visualize_partition_2d(group_id):
    h, w = group_id.shape[:2]
    output = np.zeros((h,w,3), dtype=np.uint8)
    
    num_groups = group_id.max() + 1
    group_colors = np.random.rand(num_groups, 3)
    # group_colors = np.vstack((group_colors, np.array([0,0,0])))
    
    for id in range(num_groups):
        output[np.where(group_id==id)] = group_colors[id]*255
    output[np.where(group_id==-1)] = np.array([0,0,0])
    img = Image.fromarray(output)
    img.save('sam_mask.png')
    return output

def num_to_natural(group_ids):
    '''
    Change the group number to natural number arrangement
    '''
    if np.all(group_ids == -1):
        return group_ids
    array = copy.deepcopy(group_ids)
    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return array

def visualize_2d(img_color, labels, img_size, save_path):
    import matplotlib.pyplot as plt
    # from skimage.segmentation import mark_boundaries
    # from skimage.color import label2rgb
    label_names = ["wall", "floor", "cabinet", "bed", "chair",
        "sofa", "table", "door", "window", "bookshelf",
        "picture", "counter", "desk", "curtain", "refridgerator",
        "shower curtain", "toilet", "sink", "bathtub", "other"]
    SCANNET_COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                    6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                    13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}

    colors = np.array(list(SCANNET_COLOR_MAP_20.values()))[1:]

    # import matplotlib.patches as mpatches
    # patches = []
    # for i in range(20):
    #     label=label_names[i]
    #     cur_color = colors[i]/255.0
    #     red_patch = mpatches.Patch(color=cur_color, label=label)
    #     patches.append(red_patch)
    # plt.figure()
    # plt.axis('off')
    # legend = plt.legend(frameon=False, handles=patches, loc='lower left', ncol=7, bbox_to_anchor=(0, -0.3), prop={'size': 5}, handlelength=0.7)
    # fig  = legend.figure
    # fig.canvas.draw()
    # bbox  = legend.get_window_extent()
    # bbox = bbox.from_extents(*(bbox.extents + np.array([-5,-5,5,5])))
    # bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    # plt.savefig("scannet_bar", bbox_inches=bbox, dpi=300)        

    segmentation_color = np.zeros((img_size[0], img_size[1], 3))
    for i, color in enumerate(colors):
        segmentation_color[labels == i] = color
    alpha = 0.8

    overlay = (img_color * (1-alpha) + segmentation_color * alpha).astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(overlay)
    patches = [plt.plot([], [], 's', color=np.array(color)/255, label=label)[0] for label, color in zip(label_names, colors)]
    plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4, fontsize='small')
    plt.savefig(save_path)
    plt.show()

def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #!### Dataset specific parameters #####
    image_paths = []
    args.data_root_2d = '/home/liruihuang/openscene/data_nus/nuscenes_2d'
    args.data_root_3d = '/home/liruihuang/openscene/data_nus/nuscenes_3d'
    split = 'train'

    img_dim=[800,450]
    args.cut_num_pixel_boundary = 5 # do not use the features on the image boundary
    args.keep_features_in_memory = False # keep image features in the memory, very expensive
    args.feat_dim = 768 # CLIP feature dimension
    args.point2img_mapper = PointCloudToImageMapper(
              image_dim=img_dim, 
              cut_bound=args.cut_num_pixel_boundary)

    split = args.split

    process_id_range = args.process_id_range


    data_paths = sorted(glob(join(args.data_root_3d, split, '*.pth')))
    total_num = len(data_paths)

    id_range = None
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]

    # clip_file_name = 'saved_text_embeddings/clip_scannet_200_labels_768.pt'
    # args.text_features = torch.load(clip_file_name).cpu()

    for i in trange(total_num):
        if id_range is not None and \
           (i<id_range[0] or i>id_range[1]):
            print('skip ', i, data_paths[i])
            continue
        
        process_one_scene(data_paths[i], args)



if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)
    main(args)

