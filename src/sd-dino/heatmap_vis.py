import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

from extractor_sd import load_model, process_features_and_mask
from utils.utils_correspondence import co_pca, resize 
from extractor_dino import ViTExtractor

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Initialize the argument parser
parser = argparse.ArgumentParser(description='Process the source and target image paths.')

# Define arguments
parser.add_argument('src_img_path', type=str, help='Path to the source image')
parser.add_argument('trg_img_path', type=str, help='Path to the target image')

# Define an argument for selecting the similarity function
parser.add_argument('similarity_func', type=str, choices=['l2', 'cos'], help='Similarity function to use: "l2_dist" for L2 distance or "cos" for cosine similarity')
# Define an argument for selecting the similarity function
parser.add_argument('dinov2_size', type=str, choices=['small', 'base'], help='Type of dino v2 model to use.')


# Parse arguments
args = parser.parse_args()

# Use the provided arguments
src_img_path = args.src_img_path
trg_img_path = args.trg_img_path

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
MODEL_SIZE = args.dinov2_size# # 'small' or 'base', indicate dinov2 model
TEXT_INPUT = False
SEED = 42
TIMESTEP = 100 #flexible from 0~200

# DIST = 'l2' if FUSE_DINO and not ONLY_DINO else 'cos'; not needed only using cosine similarity at the moment, no l2 distance
if ONLY_DINO:
    FUSE_DINO = True

# Seed initialization for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

# Load the stable diffusion model
if not ONLY_DINO:
    model, aug = load_model(diffusion_ver=VER, image_size=SIZE, num_timesteps=TIMESTEP)
else:
    model = None
    aug = None

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
def compute_pair_feature(model, aug, files, category, mask=False, real_size=960, dist = args.similarity_func):
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

    N = len(files) // 2
    pbar = tqdm(total=N)
    result = []
    for pair_idx in range(N):

        # Load image 1
        img1 = Image.open(files[2*pair_idx]).convert('RGB')
        img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
        img1 = resize(img1, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

        # Load image 2
        img2 = Image.open(files[2*pair_idx+1]).convert('RGB')
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

# Category for the images, and pair them
category = "pets"
files = [src_img_path, trg_img_path]

# Compute the features for the image pair
result = compute_pair_feature(model, aug, files, mask=MASK, category=category)

# Reshape the features for visualization
feature1 = result[0][0]
feature2 = result[0][1]
src_feature_reshaped = feature1.squeeze().permute(1,0).reshape(1,-1,60,60).cuda()
tgt_feature_reshaped = feature2.squeeze().permute(1,0).reshape(1,-1,60,60).cuda()
# Upsample the features for higher resolution
src_feature_upsampled = F.interpolate(src_feature_reshaped, size=(RESOLUTION, RESOLUTION), mode='bilinear')
tgt_feature_upsampled = F.interpolate(tgt_feature_reshaped, size=(RESOLUTION, RESOLUTION), mode='bilinear')

# Compute volume of cosine similarity
if args.similarity_func == 'cos':
    volume = cosine_similarity(src_feature_upsampled,tgt_feature_upsampled)
elif args.similarity_func == 'l2':
    volume = l2_distance(src_feature_upsampled,tgt_feature_upsampled)

# Load and process source and target images for visualization
src_img=Image.open(src_img_path).convert('RGB')
tgt_img=Image.open(trg_img_path).convert('RGB')
src_img = resize(src_img, RESOLUTION, resize=True, to_pil=False, edge=EDGE_PAD)
tgt_img = resize(tgt_img, RESOLUTION, resize=True, to_pil=False, edge=EDGE_PAD)

# Extract the similarity tensor
similarity_tensor = volume.squeeze()

# Initialize the plot for heatmap visualization
fig, (ax1, ax2) = plt.subplots(1, 2)

# Plot the first image in the first subplot
ax1.imshow(src_img.squeeze())

# Initialize the second subplot with an empty image
ax2.imshow(tgt_img.squeeze(), cmap='viridis')

# Create a divider to adjust the size of the second subplot
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(cm.ScalarMappable(cmap='jet'), cax=cax)
cbar.set_label('Correlation Value')

# Initialize a variable to store the previous highlighted pixel position
prev_highlighted_pixel = None

# Function to handle hover events for interactive visualization
def on_hover(event):
    global prev_highlighted_pixel
    global src_img

    if event.inaxes == ax1:
        if event.xdata is not None and event.ydata is not None:
            
            x, y = int(event.xdata), int(event.ydata)
            
            heatmap = similarity_tensor[y, x, :, :].cpu().numpy()

            # Find the nearest neighbor
            if args.similarity_func == 'cos':
                index = np.argmax(heatmap)
            elif args.similarity_func == 'l2':
                index = np.argmin(heatmap)
            x_nn = index // RESOLUTION
            y_nn = index % RESOLUTION
            
            # Clear the second subplot
            ax2.clear()
            ax2.imshow(tgt_img.squeeze())  # Display the second image
            ax2.imshow(heatmap, cmap='jet', alpha=0.6)  # Overlay the heatmap
            ax2.plot(y_nn, x_nn, 'rx', markersize=10)

            # Highlight the hovered pixel with a red crosshair
            ax1.clear()
            ax1.imshow(src_img.squeeze())
            ax1.plot(x, y, 'rx', markersize=10)
            
            prev_highlighted_pixel = (y, x)  # Update the previously highlighted pixel
            
    else:
        # Clear ax2 and display only tgt_img if mouse is outside src_img
        ax2.clear()
        ax2.imshow(tgt_img.squeeze())

    fig.canvas.draw()

# Connect the hover event to the figure
fig.canvas.mpl_connect('motion_notify_event', on_hover)

# Display the plot
plt.show()