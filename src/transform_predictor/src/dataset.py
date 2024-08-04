import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
import json

class FlowImageDataset():
    def __init__(self, flow_image_path, poses_path, seed=42, subset=None, transform=None):
        for path in [flow_image_path, poses_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} does not exist.")
        self.transform = transform
        self.flow_images, flow_image_names = self.load_and_preprocess_flow_images(flow_image_path)
        
        np.random.seed(seed)
        random_idx = np.arange(self.flow_images.shape[0])
        np.random.shuffle(random_idx)
        if subset == "train":
            train_idx = random_idx[:int(0.8*len(random_idx))]
            self.flow_images = self.flow_images[train_idx]
            self.flow_image_names = [flow_image_names[i] for i in train_idx]
        elif subset == "val":
            val_idx = random_idx[int(0.8*len(random_idx)):]
            self.flow_images = self.flow_images[val_idx]
            self.flow_image_names = [flow_image_names[i] for i in val_idx]
        elif subset == "test":
            self.flow_images = self.flow_images
            self.flow_image_names = flow_image_names
        
        poses = self.load_poses(poses_path)
        # compute the transformations for the camera s.t. it sees the object from the same viewpoint
        # as the camera in the reference image
        transforms = self.compute_transforms(poses)

        self.y = torch.tensor(transforms, dtype=torch.float32)
        print(self.y.shape)

    def compute_transforms(self, poses):
        transforms = np.empty((len(self.flow_image_names), 4, 4))
        for idx, image_name in tqdm(enumerate(self.flow_image_names)):
            obj_src = image_name.split("__")[0]
            obj_ref = image_name.split("__")[1]
            cam_id_src = obj_src.split("_")[-1]
            object_name_src = obj_src[:-len(cam_id_src)-1]
            cam_id_ref = obj_ref.split("_")[-1]
            object_name_ref = obj_ref[:-len(cam_id_ref)-1]

            obj_src_pose = poses[object_name_src][object_name_src]
            cam_src_pose = poses[object_name_src]['cam_'+str(cam_id_src)]
            cam_to_obj_src = np.linalg.inv(cam_src_pose) @ obj_src_pose

            obj_ref_pose = poses[object_name_ref][object_name_ref]
            cam_ref_pose = poses[object_name_ref]['cam_'+str(cam_id_ref)]
            cam_to_obj_ref = np.linalg.inv(cam_ref_pose) @ obj_ref_pose

            transform = cam_to_obj_src @ np.linalg.inv(cam_to_obj_ref)
            transforms[idx] = transform
        return transforms

    def load_poses(self, poses_path):
        poses = {}
        for folder in sorted(os.listdir(poses_path)):
            if 'flow_images' in folder:
                continue
            with open(os.path.join(poses_path, folder, 'poses.json')) as f:
                pose = json.load(f)
            poses[folder] = pose
        return poses

    def load_and_preprocess_flow_images(self, flow_image_path):
        flow_images = []
        flow_image_names = []
        for file in tqdm(sorted(os.listdir(flow_image_path))):
            if not file.endswith(".npy"):
                raise ValueError(f"File {file} is not a .npy file.")
            flow_image = np.load(os.path.join(flow_image_path, file)).astype(np.float32)
            height, width, _ = flow_image.shape
            # convert flow image (u, v) to hsv image with saturation = 255
            hsv = np.zeros((height, width, 3), dtype=np.uint8)
            hsv[..., 1] = 255
            mag, ang = cv2.cartToPolar(flow_image[..., 0], flow_image[..., 1])
            hsv[..., 0] = (ang * 180 / np.pi / 2)
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            flow_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            flow_image = np.transpose(flow_image, (2, 0, 1))
            flow_image = torch.tensor(flow_image, dtype=torch.float32).unsqueeze(0)
            flow_images.append(flow_image)
            flow_image_names.append(file[:-4])
        flow_images = torch.cat(flow_images, 0)
        return flow_images, flow_image_names

    def __len__(self):
        return self.flow_images.shape[0]
    
    def __getitem__(self, idx):
        flow_image = self.flow_images[idx]
        if self.transform is not None:
            flow_image = self.transform(flow_image)
        y_transl = self.y[idx, :3, 3]
        y_rot = self.y[idx, :3, :3]
        # return x, y, z, followed by the 9 elements of the rotation matrix
        y = torch.cat([y_transl, y_rot.flatten()])
        return flow_image, y