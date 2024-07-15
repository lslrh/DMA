import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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
    parser.add_argument('--data_dir', type=str, default='./data/', help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='train', help='split: "train"| "val"')
    parser.add_argument('--output_dir', type=str, default='./output', help='Where is the base logging directory')
    parser.add_argument('--openseg_model', type=str, default='./openseg_exported_clip', help='Where is the exported OpenSeg model')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--img_feat_dir', type=str, default='', help='the id range to process')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(data_path, out_dir, args):
    '''Process one scene.'''
    # short hand
    scene_id = data_path.split('/')[-1].split('_vh')[0]

    num_rand_file_per_scene = args.num_rand_file_per_scene
    feat_dim = args.feat_dim
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale
    openseg_model = args.openseg_model
    text_emb = args.text_emb
    keep_features_in_memory = args.keep_features_in_memory

    # load 3D data (point cloud)
    locs_in = torch.load(data_path)[0]
    n_points = locs_in.shape[0]

    n_interval = num_rand_file_per_scene
    n_finished = 0
    for n in range(n_interval):

        if exists(join(out_dir, scene_id +'_%d.pt'%(n))):
            n_finished += 1
            print(scene_id +'_%d.pt'%(n) + ' already done!')
            continue
    if n_finished == n_interval:
        return 1

    # short hand for processing 2D features
    scene = join(args.data_root_2d, scene_id)
    img_dirs = sorted(glob(join(scene, 'color/*')), key=lambda x: int(os.path.basename(x)[:-4]))
    num_img = len(img_dirs)
    device = torch.device('cpu')
    
    # load the text descriptions and extract the features
    # labelset = list(np.load(join(scene, 'categories.npy')))
    # labelset.append('other')
    labelset = ['A white microwave is sitting on top of a counter next to a white refrigerator.']

    args.text_features = extract_clip_feature(labelset, model_name="ViT-L/14@336px").cpu()
    # torch.save(args.text_features, join(scene, 'clip_openseg_labels.pt'))

    # extract image features and keep them in the memory
    # default: False (extract image on the fly)
    if keep_features_in_memory and openseg_model is not None:
        img_features = []
        for img_dir in tqdm(img_dirs):
            img_features.append(extract_openseg_img_feature(img_dir, openseg_model, text_emb, img_size=[240, 320]))

    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    sum_features = torch.zeros((n_points_cur, feat_dim), device=device)
    n_classes = len(args.text_features)
    pred_cls_num = torch.zeros((n_points_cur,  ), device=device)

    ################ Feature Fusion ###################
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)

    for img_id, img_dir in enumerate(tqdm([img_dirs[0]])):
        img_id = 241
        img_dir = img_dir.replace('0.jpg', '4820.jpg')

        # load pose
        posepath = img_dir.replace('color', 'pose').replace('.jpg', '.txt')
        pose = np.loadtxt(posepath)

        # load depth and convert to meter
        depth = imageio.v2.imread(img_dir.replace('color', 'depth').replace('jpg', 'png')) / depth_scale

        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        vis_id[:, img_id] = mask
        if keep_features_in_memory:
            feat_2d = img_features[img_id].to(device)
        else:
            feat_2d = extract_openseg_img_feature(img_dir, openseg_model, text_emb, img_size=[240, 320]).to(device)

        feat_2d = F.normalize(feat_2d.float().permute(1, 2, 0), 2, 2, eps=1e-5).unsqueeze(2)

        feat_similarity = torch.sigmoid(torch.matmul(feat_2d, args.text_features.float().T)/0.1).squeeze(2).squeeze(2)
        semantic_mask = (feat_similarity > 0.80).int()
        # feat_similarity = F.softmax(torch.matmul(feat_2d, args.text_features.float().T)/0.1, dim=3).squeeze(2)
        # value, semantic_mask = torch.max(feat_similarity, dim=2)

        sam_mask = np.array(Image.open(img_dir.replace('scannet_2d', 'scannet_2d_paint').replace('color/', '').replace('jpg', 'png')), dtype=np.int16)
        sam_mask = num_to_natural(F.interpolate(torch.from_numpy(sam_mask).unsqueeze(0).unsqueeze(1).float(), scale_factor=0.5).int().squeeze().numpy())
        
        # semantic_sam_mask = semantic_mask.clone()
        # num_sam_mask = len(np.unique(sam_mask))
        # for i in range(num_sam_mask):
        #     cls_tmp, cls_num = np.unique(semantic_mask[sam_mask==i], return_counts=True)
        #     # print(f'cls_num of the {i}-th mask is {cls_num}')
        #     if len(cls_num)>0:
        #         semantic_sam_mask[sam_mask==i] = cls_tmp[np.argmax(cls_num)]            
        # label_one_hot = F.one_hot(semantic_mask, n_classes)
        # img = imageio.v2.imread(img_dir)
        # visualize_partition_2d(semantic_mask.numpy(), img, './semantic_mask.png')
        # visualize_partition_2d(sam_mask, img, './sam_mask.png')
        # visualize_partition_2d(semantic_sam_mask, img, './semantic_sam_mask.png')
        # visualize_2d(img, semantic_sam_mask*(semantic_sam_mask==6), semantic_mask.shape, './semantic_sam_mask.png')

        label_2d_3d = semantic_mask[mapping[:, 1], mapping[:, 2]] 
        # pred_cls_num[mask!=0] += label_2d_3d[mask!=0]

        pred_cls_num[mask!=0] = label_2d_3d[mask!=0].float()
        # feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0)

        # counter[mask!=0]+= 1
        # sum_features[mask!=0] += feat_2d_3d[mask!=0]
    # value, label_3d = torch.max(pred_cls_num, dim=-1)
    # label_3d[value==0] = -1
    
    # save_path = scene.replace('scannet_2d', 'scannet_3d_ram_sam_paint/train') + '.pth'
    # torch.save(label_3d, save_path)
    visualize_partition(locs_in, pred_cls_num.int(), './'+img_dir.split('/')[-1].replace('.jpg', '.pc.ply'))
    import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    # counter[counter==0] = 1e-5
    # feat_bank = sum_features/counter
    # point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

    # save_fused_feature(feat_bank, point_ids, n_points, out_dir, scene_id, args)

def visualize_partition(coord, group_id, save_path):
    # SCANNET_COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
    #                 6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
    #                 13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}
    SCANNET_COLOR_MAP_20 = {0: (190., 113., 108.), 1: (88., 70., 247.)}
    colors = np.array(list(SCANNET_COLOR_MAP_20.values()))/255.0

    num_groups = group_id.max().int() + 1

    # group_colors = np.random.rand(num_groups, 3)
    segmentation_color = np.zeros((len(coord), 3))
    for i, color in enumerate(colors):
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

def visualize_partition_2d(group_id, img_color, save_path):
    h, w = group_id.shape[:2]
    output = np.zeros((h,w,3), dtype=np.uint8)
    
    num_groups = group_id.max() + 1
    np.random.seed(1024)
    group_colors = np.random.rand(num_groups, 3)
    # group_colors = np.vstack((group_colors, np.array([0,0,0])))

    for id in range(num_groups):
        if id==0:
            output[np.where(group_id==id)] = group_colors[id]*0
        else:
            output[np.where(group_id==id)] = group_colors[id]*255
    output[np.where(group_id==-1)] = np.array([0,0,0])
    
    alpha = 0.5
    img = (img_color * (1-alpha) + output * alpha).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(save_path)
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
    img_dim = (320, 240)
    depth_scale = 1000.0
    #######################################
    visibility_threshold = 0.25 # threshold for the visibility check

    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = 10 # do not use the features on the image boundary
    args.keep_features_in_memory = False # keep image features in the memory, very expensive
    args.feat_dim = 768 # CLIP feature dimension

    split = args.split
    data_dir = args.data_dir

    data_root = join(data_dir, 'scannet_3d')
    data_root_2d = join(data_dir,'scannet_2d')
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

    # load the openseg model
    saved_model_path = args.openseg_model
    args.text_emb = None
    if args.openseg_model != '':
        args.openseg_model = tf2.saved_model.load(saved_model_path,
                    tags=[tf.saved_model.tag_constants.SERVING],)
        args.text_emb = tf.zeros([1, 1, args.feat_dim])
    else:
        args.openseg_model = None

    # load intrinsic parameter
    intrinsics=np.loadtxt(os.path.join(args.data_root_2d, 'intrinsics.txt'))

    # calculate image pixel-3D points correspondances
    args.point2img_mapper = PointCloudToImageMapper(
            image_dim=img_dim, intrinsics=intrinsics,
            visibility_threshold=visibility_threshold,
            cut_bound=args.cut_num_pixel_boundary)

    data_paths = sorted(glob(join(data_root, split, '*.pth')))
    total_num = len(data_paths)

    id_range = None
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]

    # clip_file_name = 'saved_text_embeddings/clip_scannet_labels_768.pt'
    # args.text_features = torch.load(clip_file_name).cpu()
    for i in trange(total_num):
        i=1
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

