import os
import torch
import numpy as np
from glob import glob
from os.path import join, exists
from tqdm import tqdm, trange
from util.util import export_pointcloud
from plyfile import PlyData, PlyElement

# data_paths = './data_matterport/matterport_multiview_openseg_test/pa4otMbVnkk_region8_0.pt'
# scene_name = 'YVUC4YcDtcY'
# file_dirs = sorted(glob(join('./data_matterport/matterport_3d/test', scene_name + '_*.pth')))
# pcl_list=[]
# color_list = []

# for file in file_dirs:
#     pc = torch.load(file)
#     pcl_list.append(pc[0])
#     color_list.append(pc[1])

# pcl = np.concatenate(pcl_list, axis=0)
# color = np.concatenate(color_list, axis=0)
# # import pdb; pdb.set_trace()

# mask = pcl[:,2]<4.0

# pcl = pcl[mask]
# color = color[mask]

# color = (color+1)/2
# save_path = join('./data_matterport/visualization', scene_name)
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
pc = torch.load('./data/scannet_3d/train/scene0464_00_vh_clean_2.pth')
pcl=pc[0]
color = (pc[1]+1)/2

export_pointcloud('0464.ply', pcl, colors=color)
import pdb; pdb.set_trace()
# file_dirs_1 = [file.replace('.pth', '_distill.npy').replace('data_matterport/matterport_3d/test','save_matter_openscene') for file in file_dirs]
# # file_dirs = sorted(glob(join('./save_matter_openscene', scene_name + '_*_distill.npy')))
# color_list = []

# for file in file_dirs_1:
#     # plydata = PlyData.read(file)
#     # data = plydata.elements[0].data
#     color = torch.load(file)
#     # color = np.array([[r, g, b] for _,_,_,r,g,b in data])
#     # pcl = np.array([[x, y, z] for x,y,z,r,g,b in data])
#     color_list.append(color)
#     # pcl_list.append(pcl)

# # pcl = np.concatenate(pcl_list, axis=0)
# color = np.concatenate(color_list, axis=0)
# color = color[mask]
# # color = (color+1)/2
# export_pointcloud(join(save_path, 'openscene_distill.ply'), pcl, colors=color)

# # file_dirs = sorted(glob(join('./save_matter_openscene', scene_name + '_*_gt.npy')))
# file_dirs_1 = [file.replace('.pth', '_gt.npy').replace('data_matterport/matterport_3d/test','save_matter_openscene') for file in file_dirs]
# color_list = []

# for file in file_dirs_1:
#     # plydata = PlyData.read(file)
#     # data = plydata.elements[0].data
#     color = torch.load(file)
#     # color = np.array([[r, g, b] for _,_,_,r,g,b in data])
#     # pcl = np.array([[x, y, z] for x,y,z,r,g,b in data])
#     color_list.append(color)
#     # pcl_list.append(pcl)

# # pcl = np.concatenate(pcl_list, axis=0)
# color = np.concatenate(color_list, axis=0)
# color = color[mask]
# export_pointcloud(join(save_path, 'gt.ply'), pcl, colors=color)


# # file_dirs = sorted(glob(join('./save_matter_dma', scene_name + '_*_distill.npy')))
# file_dirs_1 = [file.replace('.pth', '_distill.npy').replace('data_matterport/matterport_3d/test','save_matter_dma') for file in file_dirs]
# color_list = []

# for file in file_dirs_1:
#     # plydata = PlyData.read(file)
#     # data = plydata.elements[0].data
#     color = torch.load(file)
#     # color = np.array([[r, g, b] for _,_,_,r,g,b in data])
#     # pcl = np.array([[x, y, z] for x,y,z,r,g,b in data])
#     color_list.append(color)
#     # pcl_list.append(pcl)

# # pcl = np.concatenate(pcl_list, axis=0)
# color = np.concatenate(color_list, axis=0)
# color = color[mask]
# export_pointcloud(join(save_path, 'dma_distill.ply'), pcl, colors=color)

# # file_dirs = sorted(glob(join('./save_mattertext_only', scene_name + '_*_distill.npy')))
# file_dirs_1 = [file.replace('.pth', '_distill.npy').replace('data_matterport/matterport_3d/test','save_mattertext_only') for file in file_dirs]
# color_list = []

# for file in file_dirs_1:
#     # plydata = PlyData.read(file)
#     # data = plydata.elements[0].data
#     color = torch.load(file)
#     # color = np.array([[r, g, b] for _,_,_,r,g,b in data])
#     # pcl = np.array([[x, y, z] for x,y,z,r,g,b in data])
#     color_list.append(color)
#     # pcl_list.append(pcl)

# # pcl = np.concatenate(pcl_list, axis=0)
# color = np.concatenate(color_list, axis=0)
# color = color[mask]
# export_pointcloud(join(save_path, 'text_distill.ply'), pcl, colors=color)


# data_paths = sorted(glob(join('./data_nus/nuscenes_2d/nuscenes_multiview_openseg/', '*.pt')))
# for i in trange(len(data_paths)):
#     try:
#         A = torch.load(data_paths[i])
#     except Exception as ex:
#         print(data_paths[i])
#         import pdb; pdb.set_trace()
# # A = np.load('./data/tags_tag2text/scene0000_00.npy')
# import pdb; pdb.set_trace()
# processed_data = torch.load('./data_nus/nuscenes_2d/nuscenes_multiview_fcclip/cb57adad69c54307907d522fd78c543b.pt')
# import pdb; pdb.set_trace()