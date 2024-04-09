import sys
import argparse
import os
import time
import logging
from datetime import datetime
import torch
import json
from pathlib import Path
import cv2
import torchvision
from tqdm import tqdm
import trimesh
import numpy as np
import open3d as o3d
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file', default='vanilla')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--camera', '-c', default=None, help='path to cameras.json')
    parser.add_argument('--model', '-m', default=None, help='path to the model')
    parser.add_argument('--output-dir', '-o', default=None, help='path to the output dir')
    parser.add_argument('--load_iteration', default=-1, type=int, help='iteration to be rendered')
    parser.add_argument('--resolution', default=2, type=int, help='downscale resolution')
    parser.add_argument('--sh', default=0, type=int, help='default SH degree')
    parser.add_argument('--white_background', action='store_true', help='use white background')
    parser.add_argument('--clean', action='store_true', help='perform a clean operation')
    args, extras = parser.parse_known_args()
    
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))
    
    from gaustudio.utils.misc import load_config
    from gaustudio import models, datasets, renderers
    from gaustudio.datasets.utils import JSON_to_camera
    from gaustudio.utils.graphics_utils import depth2point
    # parse YAML config to OmegaConf
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, '../configs', args.config+'.yaml')
    config = load_config(config_path, cli_args=extras)
    config.cmd_args = vars(args)  
    
    pcd = models.make(config.model.pointcloud)
    renderer = renderers.make(config.renderer)
    pcd.active_sh_degree = args.sh
    
    model_path = args.model
    if os.path.isdir(model_path):
        if args.load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(args.model, "point_cloud"))
        else:
            loaded_iter = args.load_iteration
        work_dir = os.path.join(model_path, "renders", "iteration_{}".format(loaded_iter)) if args.output_dir is None else args.output_dir
        
        print("Loading trained model at iteration {}".format(loaded_iter))
        pcd.load(os.path.join(args.model,"point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"))
    elif model_path.endswith(".ply"):
        work_dir = os.path.join(os.path.dirname(model_path), os.path.basename(model_path)[:-4]) if args.output_dir is None else args.output_dir
        pcd.load(model_path)
    else:
        print("Model not found at {}".format(model_path))
    pcd.to("cuda")
    
    if args.camera is None:
        args.camera = os.path.join(model_path, "cameras.json")
    if os.path.exists(args.camera):
        print("Loading camera data from {}".format(args.camera))
        with open(args.camera, 'r') as f:
            camera_data = json.load(f)
        cameras = []
        for camera_json in camera_data:
            camera = JSON_to_camera(camera_json, "cuda")
            cameras.append(camera)
        o3d_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.02,
            sdf_trunc=0.08,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
    else:
        assert "Camera data not found at {}".format(args.camera)

    bg_color = [1,1,1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print("Fusing mesh...")
    for camera in tqdm(cameras[::3]):
        camera.downsample_scale(args.resolution)
        camera = camera.to("cuda")
        with torch.no_grad():
            render_pkg = renderer.render(camera, pcd)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        rendered_depth = render_pkg["rendered_median_depth"][0]
        invalid_mask = rendered_depth == 15.0

        rendering[:, invalid_mask] = 0.
        rendered_depth[invalid_mask] = 0

        intrinsic = camera.intrinsics.cpu().numpy()
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        paspect = fy / fx
        width, height = camera.image_width, camera.image_height
        
        rendered_depth_o3d = o3d.geometry.Image(rendered_depth.cpu().numpy())
        rendered_rgb_np = np.asarray(rendering.permute(1,2,0).cpu().numpy(), order="C")
        rendered_rgb_o3d = o3d.geometry.Image((rendered_rgb_np*255).astype(np.uint8))
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rendered_rgb_o3d, rendered_depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
        )
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic_np = camera.extrinsics.cpu().numpy()
        o3d_volume.integrate(rgbd_o3d, intrinsic_o3d, extrinsic_np)

    gs_mesh = o3d_volume.extract_triangle_mesh()
    gs_mesh_path = os.path.join(work_dir, 'gs_mesh.ply')

    if args.clean:
        # Clean Mesh
        clean_threshold = 0.05
        print("Analyzing connected components...")
        (triangle_clusters, 
        cluster_n_triangles,
        cluster_area) = gs_mesh.cluster_connected_triangles()
        
        print("Finding largest component...") 
        triangle_clusters = np.array(triangle_clusters)  
        cluster_n_triangles = np.array(cluster_n_triangles)
        
        largest_cluster_idx = cluster_n_triangles.argmax()
        largest_cluster_n_triangles = cluster_n_triangles[largest_cluster_idx]

        print(f"Largest component has {largest_cluster_n_triangles} triangles")
        
        triangles_keep_mask = np.zeros_like(triangle_clusters, dtype=np.int32)
        saved_clusters = []
        
        print("Removing small components...")
        for i, n_tri in enumerate(cluster_n_triangles):
            if n_tri > clean_threshold * largest_cluster_n_triangles:
                saved_clusters.append(i)
                triangles_keep_mask += triangle_clusters == i
                
        triangles_to_remove = triangles_keep_mask == 0
        gs_mesh.remove_triangles_by_mask(triangles_to_remove)
        gs_mesh.remove_unreferenced_vertices()
        
        print(f"Removed {triangles_to_remove.sum()} triangles")
    print(f"Saving processed mesh to {gs_mesh_path}") 
    o3d.io.write_triangle_mesh(gs_mesh_path, gs_mesh)

if __name__ == '__main__':
    main()
