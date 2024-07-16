import os 
import numpy as np
import json
import h5py
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import cv2

from extractor_sd import load_model, process_features_and_mask
from utils.utils_correspondence import co_pca, resize 
from extractor_dino import ViTExtractor

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.my_utils.load_data import load_data

# Configuration parameters
MASK = False
VER = "v1-5"
PCA = False
CO_PCA = True
PCA_DIMS = [256, 256, 256]
SIZE =960
RESOLUTION = 128
EDGE_PAD = False

FUSE_DINO = 1
ONLY_DINO = 1
DINOV2 = True
MODEL_SIZE = 'small'# # 'small' or 'base', indicate dinov2 model
TEXT_INPUT = False
SEED = 42
TIMESTEP = 100 #flexible from 0~200
DIST = 'cos' # 'cos' or 'l2' or 'l1'

# DIST = 'l2' if FUSE_DINO and not ONLY_DINO else 'cos'; not needed only using cosine similarity at the moment, no l2 distance
if ONLY_DINO:
    FUSE_DINO = True

# Seed initialization for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True


def visualize_data(rgb_images, object_masks):
    for rgb, object_mask in zip(rgb_images, object_masks):
        plt.subplot(1, 2, 1)
        plt.imshow(rgb)
        plt.subplot(1, 2, 2)
        plt.imshow(object_mask)
        plt.show()

# Function to calculate cosine similarity between feature maps
def cosine_similarity(features1, features2):

    # Flatten the features map
    fm1_flat = features1.view(features1.shape[0], features1.shape[1], -1).permute(0, 2, 1)
    fm2_flat = features2.view(features2.shape[0], features2.shape[1], -1).permute(0, 2, 1)

    # Normalize the feature maps
    fm1_norm = F.normalize(fm1_flat, p=2, dim=2)
    fm2_norm = F.normalize(fm2_flat, p=2, dim=2)

    # Compute cosine similarity
    cosine_sim = torch.matmul(fm1_norm, fm2_norm.transpose(1, 2))

    # Reshape to [batch_size, height, width, height, width]
    cosine_sim = cosine_sim.view(features1.shape[0], features1.shape[2], features1.shape[3], features2.shape[2], features2.shape[3])

    return cosine_sim

# Function to calculate l2 distance between feature maps
def l2_distance(features1, features2):
    # Flatten the feature maps
    fm1_flat = features1.view(features1.shape[0], features1.shape[1], -1).permute(0, 2, 1)  # [B, H1*W1, C]
    fm2_flat = features2.view(features2.shape[0], features2.shape[1], -1)  # [B, C, H2*W2]

    # Compute the norms squared
    fm1_norm2 = torch.sum(fm1_flat ** 2, dim=2, keepdim=True)  # [B, H1*W1, 1]
    fm2_norm2 = torch.sum(fm2_flat ** 2, dim=1, keepdim=True)  # [B, 1, H2*W2]

    # Compute the dot product
    dot_product = torch.bmm(fm1_flat, fm2_flat)  # [B, H1*W1, H2*W2]

    # Compute squared L2 distance
    l2_dist_squared = fm1_norm2 + fm2_norm2.transpose(1, 2) - 2 * dot_product  # [B, H1*W1, H2*W2]

    # Taking the square root gives the L2 distance, ensure non-negative distances
    l2_dist = torch.sqrt(torch.relu(l2_dist_squared) + 1e-6)

    # Reshape to [batch_size, height1, width1, height2, width2]
    l2_dist = l2_dist.view(features1.shape[0], features1.shape[2], features1.shape[3], features2.shape[2], features2.shape[3])

    return l2_dist

# Function to compute features for a pair of images
def compute_pair_feature(model, aug, image_pairs=[], category="an object", mask=False, real_size=960, dist = DIST):
    if type(category) == str:
        category = [category]
    img_size = 840 if DINOV2 else 244
    model_dict={'small':'dinov2_vits14',
                'base':'dinov2_vitb14',
                'large':'dinov2_vitl14',
                'giant':'dinov2_vitg14'}
    
    model_type = model_dict[MODEL_SIZE] if DINOV2 else 'dino_vits8'
    layer = 11 if DINOV2 else 9
    if 'l' in model_type:
        layer = 23
    elif 'g' in model_type:
        layer = 39
    facet = 'token' if DINOV2 else 'key'
    stride = 14 if DINOV2 else 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # indiactor = 'v2' if DINOV2 else 'v1'
    # model_size = model_type.split('vit')[-1]
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size[0] if DINOV2 else extractor.model.patch_embed.patch_size
    num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)
    
    input_text = "a photo of "+category[-1][0] if TEXT_INPUT else None

    pbar = tqdm(total=len(image_pairs))
    result = []
    for pair_idx in range(len(image_pairs)):

        # Load image 1
        img1 = Image.fromarray(image_pairs[pair_idx][0])
        img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img1 = resize(img1, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

        # Load image 2
        img2 = Image.fromarray(image_pairs[pair_idx][1])
        img2_input = resize(img2, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img2 = resize(img2, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

        with torch.no_grad():
            if not CO_PCA:
                if not ONLY_DINO:
                    img1_desc = process_features_and_mask(model, aug, img1_input, input_text=input_text, mask=False, pca=PCA).reshape(1,1,-1, num_patches**2).permute(0,1,3,2)
                    img2_desc = process_features_and_mask(model, aug, img2_input, category[-1], input_text=input_text,  mask=mask, pca=PCA).reshape(1,1,-1, num_patches**2).permute(0,1,3,2)
                if FUSE_DINO:
                    img1_batch = extractor.preprocess_pil(img1)
                    img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)
                    img2_batch = extractor.preprocess_pil(img2)
                    img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet)

            else:
                if not ONLY_DINO:
                    features1 = process_features_and_mask(model, aug, img1_input, input_text=input_text,  mask=False, raw=True)
                    features2 = process_features_and_mask(model, aug, img2_input, input_text=input_text,  mask=False, raw=True)
                    processed_features1, processed_features2 = co_pca(features1, features2, PCA_DIMS)
                    img1_desc = processed_features1.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
                    img2_desc = processed_features2.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
                if FUSE_DINO:
                    img1_batch = extractor.preprocess_pil(img1)
                    img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)
                    img2_batch = extractor.preprocess_pil(img2)
                    img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet)
                
            if dist == 'l1' or dist == 'l2':
                # normalize the features
                if not ONLY_DINO:
                    img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                    img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
                if FUSE_DINO:
                    img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
                    img2_desc_dino = img2_desc_dino / img2_desc_dino.norm(dim=-1, keepdim=True)

            if FUSE_DINO and not ONLY_DINO:
                # cat two features together
                img1_desc = torch.cat((img1_desc, img1_desc_dino), dim=-1)
                img2_desc = torch.cat((img2_desc, img2_desc_dino), dim=-1)

            if ONLY_DINO:
                img1_desc = img1_desc_dino
                img2_desc = img2_desc_dino

            result.append([img1_desc.cpu(), img2_desc.cpu()])

        pbar.update(1)
    return result

def calculate_flow(similarity_tensor, object_mask):
    flow_field = torch.zeros(similarity_tensor.shape[0], similarity_tensor.shape[1], 2)
    for x in range(similarity_tensor.shape[1]):
        for y in range(similarity_tensor.shape[0]):
            if object_mask[y, x] == 0:
                continue
            argmax = torch.argmax(similarity_tensor[y,x, :, :])
            argmax_y = argmax // similarity_tensor.shape[2]
            argmax_x = argmax % similarity_tensor.shape[2]
            flow_y = argmax_y - y
            flow_x = argmax_x - x
            flow_field[y,x,0] = flow_y
            flow_field[y,x,1] = flow_x
    return flow_field

def calculate_flow_vectorized(similarity_tensor, object_mask):
    object_mask = torch.tensor(object_mask).to(similarity_tensor.device)
    y_indices, x_indices = torch.meshgrid(torch.arange(similarity_tensor.shape[0]), torch.arange(similarity_tensor.shape[1]))
    y_indices = y_indices.to(similarity_tensor.device)
    x_indices = x_indices.to(similarity_tensor.device)

    # Calculate argmax along the last two dimensions
    argmax_y = torch.argmax(torch.max(similarity_tensor, dim=-1).values, dim=-1)
    argmax_x = torch.argmax(torch.max(similarity_tensor, dim=-2).values, dim=-1)

    flow_y = argmax_y - y_indices
    flow_x = argmax_x - x_indices

    flow_field = torch.stack((flow_y, flow_x), dim=-1)

    # Set flow_field to zero where object_mask is zero
    flow_field = flow_field * object_mask.unsqueeze(-1)

    return flow_field

if __name__ == "__main__":
    rgb_images, object_masks, poses, scene_names = load_data(Path("../output", "dataset_rendered"))
    output_dir = Path("../output", "flow_images")
    if os.path.exists(output_dir):
        raise ValueError("Output directory exists")
    os.makedirs(output_dir)

    batch_size = 512 # batch size for how many pairs to process with one function call
    
    object_names = list(rgb_images.keys())
    image_pairs = []
    masks = []
    pair_names = []
    for idx in rgb_images[object_names[0]].keys():
        image1 = rgb_images[object_names[0]][idx]
        mask1 = object_masks[object_names[0]][idx]
        scene_name1 = scene_names[object_names[0]][idx]
        mask1 = Image.fromarray(mask1)
        mask1 = resize(mask1, RESOLUTION, resize=True, to_pil=False, edge=EDGE_PAD)
        mask1 = torch.tensor(mask1).to('cuda')
        for idx2 in rgb_images[object_names[1]].keys():
            image2 = rgb_images[object_names[1]][idx2]
            mask2 = object_masks[object_names[1]][idx2]
            scene_name2 = scene_names[object_names[1]][idx2]
            image_pairs.append([image1, image2])
            pair_names.append(scene_name1 + "__" + scene_name2)
            mask2 = Image.fromarray(mask2)
            mask2 = resize(mask2, RESOLUTION, resize=True, to_pil=False, edge=EDGE_PAD)
            mask2 = torch.tensor(mask2).to('cuda')
            masks.append([mask1, mask2])
    
    # Load the stable diffusion model
    if not ONLY_DINO:
        model, aug = load_model(diffusion_ver=VER, image_size=SIZE, num_timesteps=TIMESTEP)
    else:
        model = None
        aug = None

    # Category for the images, and pair them
    category = "shoes"

    # Compute the features for the image pair
    num_batches = len(image_pairs) // batch_size
    for batch_num in range(num_batches):
        print(f"Processing batch {batch_num+1}/{num_batches}")
        start_idx = batch_num * batch_size
        end_idx = (batch_num + 1) * batch_size
        print(f"Processing images {start_idx} to {end_idx}")
        print(f"{len(image_pairs)} image pairs, {len(masks)} masks, {len(pair_names)} pair names")
        if end_idx > len(image_pairs):
            result = compute_pair_feature(model, aug, image_pairs=image_pairs[start_idx:], mask=MASK, category=category)
            masks_batch = masks[start_idx:]
            pair_names_batch = pair_names[start_idx:]
        else:
            result = compute_pair_feature(model, aug, image_pairs=image_pairs[start_idx:end_idx], mask=MASK, category=category)
            masks_batch = masks[start_idx:end_idx]
            pair_names_batch = pair_names[start_idx:end_idx]
        for i in range(len(result)):
            feature1 = result[i][0]
            feature2 = result[i][1]
            
            src_feature_reshaped = feature1.squeeze().permute(1,0).reshape(1,-1,60,60).cuda()
            tgt_feature_reshaped = feature2.squeeze().permute(1,0).reshape(1,-1,60,60).cuda()
            # Upsample the features for higher resolution
            src_feature_upsampled = F.interpolate(src_feature_reshaped, size=(RESOLUTION, RESOLUTION), mode='bilinear')
            tgt_feature_upsampled = F.interpolate(tgt_feature_reshaped, size=(RESOLUTION, RESOLUTION), mode='bilinear')

            # Compute volume of cosine similarity
            if DIST == 'cos':
                volume = cosine_similarity(src_feature_upsampled,tgt_feature_upsampled)
            elif DIST == 'l2':
                volume = l2_distance(src_feature_upsampled,tgt_feature_upsampled)

            # Extract the similarity tensor
            similarity_tensor = volume.squeeze()
            mask0 = masks_batch[i][0]
            mask1 = masks_batch[i][1]

            similarity_tensor[mask0 == 0, :, :] = 0
            similarity_tensor[:, :, mask1 == 0] = 0


            flow = calculate_flow_vectorized(similarity_tensor, mask0).to('cpu').numpy()
            image_name = pair_names_batch[i]
            save_path = os.path.join(output_dir, image_name)
            np.save(save_path, flow)