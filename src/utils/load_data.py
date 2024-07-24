import os
import json
import numpy as np
import h5py

def load_data(path):
    rgb_images = {}
    masks = {}
    poses = {}
    names = {}
    for object_name in sorted(os.listdir(path)):
        rgb_images_obj = {}
        object_masks = {}
        scene_names = {}
        for file in sorted(os.listdir(os.path.join(path, object_name))):
            if file.endswith(".hdf5"):
                rgb, _, object_mask = load_hdf5(os.path.join(path, object_name, file), object_name)
                idx = int(file.split(".hdf5")[0])
                rgb_images_obj[idx] = rgb
                object_masks[idx] = object_mask
                scene_names[idx] = object_name + "_" + str(idx)
            if file.endswith(".json"):
                obj_poses = load_poses_json(os.path.join(path, object_name, file))
        rgb_images[object_name] = rgb_images_obj
        masks[object_name] = object_masks
        poses[object_name] = obj_poses
        names[object_name] = scene_names
    return rgb_images, masks, poses, names

def load_hdf5(filepath, object_name):
    with h5py.File(filepath) as f:
        rgb = f['colors']
        depth = f['depth']
        seg = f['instance_segmaps']
        instance_attribute_map = f['instance_attribute_maps']

        # thank blenderproc for saving the data in such a shitty way!
        metadata = json.loads(np.array(instance_attribute_map).tobytes().decode('utf-8'))
        object_id = None
        for instance in metadata:
            if instance['name'] == object_name:
                object_id = instance['idx']
                break
        if object_id is None:
            raise ValueError(f"No metadata found for object {object_name}, file: {filepath}")
        
        object_mask = np.array(seg) == object_id

        return np.array(rgb), np.array(depth), object_mask

def load_poses_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data