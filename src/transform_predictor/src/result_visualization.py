import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path


class ResultVisualization:
    def __init__(self, meshes, save_dir=Path("../../output/results")) -> None:
        self.meshes = meshes
        self.save_dir = save_dir
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        width = 640  
        height = 480  
        intrinsics = np.array([[538.391033533567, 0.0, 315.3074696331638], 
                            [0.0, 538.085452058436, 233.0483557773859], 
                            [0.0, 0.0, 1.0]], dtype=np.float64)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        self.cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    
    def visualize_results(self, dataset, rgb_images, poses, predictions, te, re):
        plt.switch_backend("agg")
        for vis_idx in tqdm(range(len(dataset.flow_image_names))):
            fig = plt.figure(figsize=(10, 10))
            obj_src, obj_ref = dataset.flow_image_names[vis_idx].split("__")
            obj1_idx = int(obj_src.split("_")[-1])
            obj2_idx = int(obj_ref.split("_")[-1])
            obj_src = obj_src[:-len(str(obj1_idx))-1]
            obj_ref = obj_ref[:-len(str(obj2_idx))-1]
            img_src = rgb_images[obj_src][obj1_idx]
            img_ref = rgb_images[obj_ref][obj2_idx]
            
            obj_src_pose = poses[obj_src][obj_src]
            cam_src_pose = poses[obj_src]['cam_'+str(obj1_idx)]
            cam_to_obj_src = np.linalg.inv(cam_src_pose) @ obj_src_pose

            transform_pred = np.eye(4)
            transform_pred[:3, :3] = predictions[vis_idx][3:].reshape(3, 3).cpu().numpy()
            transform_pred[:3, 3] = predictions[vis_idx][:3].cpu().numpy()
            
            obj_to_cam = (np.linalg.inv(cam_to_obj_src) @ transform_pred)
            transform = np.linalg.inv(obj_to_cam)
            
            model_ = self.meshes[obj_src]
            mesh = deepcopy(model_.meshes[0].mesh)
            mesh_name = model_.meshes[0].mesh_name
            mat = model_.materials[0]

            # rotation around x-axis by 180 degrees because open3d and blender have different 
            # definitions for the camera coordinate system
            rot_x_180 = np.eye(4)
            rot_x_180[1, 1] = -1
            rot_x_180[2, 2] = -1
            mesh.transform((rot_x_180 @ transform))

            self.renderer.scene.add_geometry(mesh_name, mesh, mat)
            self.renderer.setup_camera(self.cam_intrinsics, np.eye(4, dtype=np.float64))
            img = self.renderer.render_to_image()
            self.renderer.scene.remove_geometry(mesh_name)
            img_np = np.asarray(img)
            fig.suptitle(f"Image pair {vis_idx}, te: {te[vis_idx]:.2f}, re: {re[vis_idx]:.2f}")
            plt.subplot(2, 2, 1)
            plt.imshow(img_src)
            plt.title("Source")
            plt.subplot(2, 2, 2)
            plt.imshow(img_ref)
            plt.title("Reference")
            plt.subplot(2, 2, 3)
            plt.imshow(img_np)
            plt.title("Prediction")
            plt.subplot(2,2,4)
            plt.imshow(dataset.flow_images[vis_idx].squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
            plt.title("Flow image")
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/{vis_idx:05d}.jpg", dpi=100)
            plt.close()
            
