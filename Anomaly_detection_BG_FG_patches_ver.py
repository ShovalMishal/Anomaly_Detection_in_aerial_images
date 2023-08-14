import json
import math
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict

import torchvision.datasets
from PIL import Image
from mmdet.models.utils import unpack_gt_instances
import numpy as np
from matplotlib import pyplot as plt
from mmengine.runner import Runner
import torch
import copy
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmrotate.utils import register_all_modules
from mmengine.config import Config
import torch.nn.functional as F
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torchvision import models
from mmdet.registry import TASK_UTILS
from mmengine.structures import InstanceData
from mmdet.structures.bbox import HorizontalBoxes
from tqdm import tqdm
import skimage

from image_pyramid_patches_dataset import image_pyramid_patches_dataset, transform_to_imshow
from results import plot_precision_recall_curve, plot_roc_curve, plot_scores_histograms
from utils import remove_empty_folders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

images_to_use = ['P0000__1024__0___2472', 'P0002__1024__824___824', 'P0021__1024__824___1648',
                 'P0022__1024__824___4120',
                 'P0036__1024__2169___1648', 'P0063__1024__824___1648', 'P0082__1024__824___3296',
                 'P0129__1024__384___824',
                 'P0136__1024__1648___1554', 'P0158__1024__1648___824', 'P0315__1024__0___0', 'P0260__1024__0___0',
                 'P0504__1024__0___0', 'P0653__1024__0___172', 'P0871__1024__1367___0', 'P0873__1024__610___824',
                 'P0913__1024__1648___0', 'P0911__1024__11___973', 'P1055__1024__3832___3418',
                 'P1374__1024__1648___824']


def create_dataloader(cfg, mode="train"):
    dataloader_mode = mode + "_dataloader"
    test_dataloader = cfg.get(dataloader_mode)
    dataloader_cfg = copy.deepcopy(test_dataloader)
    dataloader_cfg["dataset"]["_scope_"] = "mmrotate"
    data_loader = Runner.build_dataloader(dataloader_cfg, seed=123456)
    return data_loader


def show_img(img: torch.tensor, title="", path=""):
    image_tensor = img.cpu()
    # Convert the tensor to a NumPy array
    image_np = image_tensor.numpy()
    # Transpose the dimensions if needed
    # (PyTorch tensors are typically in the channel-first format, while NumPy arrays are in channel-last format)
    image_np = np.transpose(image_np, (1, 2, 0)).astype(int)
    # Display the image using matplotlib
    rgb_image = np.flip(image_np, axis=-1)
    plt.imshow(rgb_image)
    if title:
        plt.title(title)
    plt.axis('off')
    if path:
        plt.savefig(path)
    else:
        plt.show()


# calculate bboxes indices for all pyramid levels
def calc_patches_indices(pyramid_levels: int, patch_size: int, scale_factor: float, patch_stride: float, image_size):
    patches_indices_tensor = []
    curr_image_size = image_size
    for i in range(pyramid_levels):
        scale = scale_factor ** i
        curr_scale_patches_indices = compute_patches_indices_per_scale(curr_image_size, patch_stride,
                                                                       patch_size,
                                                                       scale_factor=scale)
        curr_image_size = (math.ceil(scale_factor * curr_image_size[0]), math.ceil(scale_factor * curr_image_size[1]))
        patches_indices_tensor.append(curr_scale_patches_indices)
    patches_indices_tensor = torch.cat(patches_indices_tensor, dim=0)
    patches_horizontal_boxes = HorizontalBoxes(patches_indices_tensor)
    formatted_patches_indices = InstanceData(priors=patches_horizontal_boxes)
    return formatted_patches_indices


def compute_patches_indices_per_scale_old(image_size, patch_size, scale_factor=1):
    patch_H, patch_W = patch_size
    H, W = image_size
    patches_W, _ = divmod(W, patch_W)
    patches_H, _ = divmod(H, patch_H)
    indices = torch.tensor([(i, j) for i in range(patches_H) for j in range(patches_W)])
    top_left = indices * torch.tensor(patch_size)
    bottom_right = top_left + torch.tensor(patch_size)
    corners = torch.stack([top_left, bottom_right], dim=1).float()
    corners *= 1 / scale_factor
    return corners.int()


def compute_patches_indices_per_scale(image_size, patch_stride, patch_size, scale_factor: float = 1.0):
    stride = int(patch_size * patch_stride)
    # Create a meshgrid of indices
    grid_h, grid_w = torch.meshgrid(torch.arange(image_size[0]), torch.arange(image_size[1]))

    # Create index grids for the top-left and bottom-right corners of each patch
    indices_tl_h = grid_h[:-patch_size + 1:stride, :-patch_size + 1:stride]
    indices_tl_w = grid_w[:-patch_size + 1:stride, :-patch_size + 1:stride]
    indices_br_h = indices_tl_h + patch_size
    indices_br_w = indices_tl_w + patch_size

    # Stack the index grids
    indices = torch.stack((indices_tl_h, indices_tl_w, indices_br_h, indices_br_w), dim=0)
    indices = indices.view(4, -1).T.float()
    indices *= 1 / scale_factor
    return indices.int()


def split_images_into_patches_with_no_overlaps(images, patch_size):
    # dimensions of the input batch (B: batch size, C: channels, H: height, W: width)
    B, C, H, W = images.shape
    # shape of the puzzle piece
    piece_H, piece_W = patch_size
    # calculate the maximum height and width that fits the piece shape
    max_H = (H // piece_H) * piece_H
    max_W = (W // piece_W) * piece_W
    # truncate the images to the maximum size that fits the piece shape
    images = images[:, :, :max_H, :max_W]
    # number of pieces along each dimension
    num_pieces_H = max_H // piece_H
    num_pieces_W = max_W // piece_W
    # split the images
    splits = images.view(B, C, num_pieces_H, piece_H, num_pieces_W, piece_W)
    splits = splits.permute(0, 2, 4, 1, 3, 5).contiguous()
    splits = splits.view(B, num_pieces_H * num_pieces_W, C, piece_H, piece_W)
    return splits


def split_images_into_patches(images, patch_size, patch_stride: float):
    stride = int(patch_size * patch_stride)

    patches = images.unfold(2, patch_size, stride).unfold(3, patch_size, stride)

    # Reshape the patches tensor to the desired shape
    B, C, _, _, _, _ = patches.size()
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.contiguous().view(B, -1, C, patch_size, patch_size)

    return patches


def preprocess_patches(patches):
    resized_patches = F.interpolate(patches, size=(224, 224), mode='bilinear', align_corners=False)
    resized_patches = resized_patches / 255.0
    normalized_patches = (resized_patches - mean) / std
    return normalized_patches


def create_gaussian_pyramid(data: list, pyramid_levels, scale_factor: float):
    all_pyramids = []
    for im_ind in range(len(data)):
        curr_pyr = skimage.transform.pyramid_gaussian(data[im_ind].numpy(), channel_axis=0, downscale=1 / scale_factor,
                                                      max_layer=pyramid_levels - 1)
        all_pyramids.append(list(curr_pyr))
    pyramid_batch = []
    for level_ind in range(pyramid_levels):
        curr_level = [torch.from_numpy(np.multiply(im_pyr[level_ind], 255).clip(0, 255).astype(np.float32)) for im_pyr
                      in all_pyramids]
        pyramid_batch.append(torch.stack(curr_level, dim=0))
    return pyramid_batch


def calculate_scores_and_labels(test_features, labels, train_features, k):
    # set > 0 for foreground and 0 for background
    scores = calculate_scores(test_features.to(device), train_features.to(device), k=k)
    return scores.cpu(), labels.cpu()


def calculate_scores(test_features, train_features, k=3):
    diff = torch.cdist(test_features, train_features)
    # Sort the differences along the second dimension
    sorted_diff, indices = torch.sort(diff, dim=1)
    # Select the k closest values
    closest_values = sorted_diff[:, :k]
    scores = torch.mean(closest_values.float(), dim=1)
    return scores


def convert_oriented_representation_to_bounding_square(oriented_representation, image_shape):
    h = oriented_representation.heights
    w = oriented_representation.widths
    # Correction of deviation of an object from the image
    c1 = oriented_representation.centers[:, 0] - w / 2 < 0
    c2 = oriented_representation.centers[:, 1] - h / 2 < 0
    oriented_representation.centers[c1, 0] = (w[c1] / 2 + oriented_representation.centers[c1, 0]) / 2
    oriented_representation.centers[c2, 1] = (h[c2] / 2 + oriented_representation.centers[c2, 1]) / 2
    w[c1] = oriented_representation.centers[c1, 0]
    h[c2] = oriented_representation.centers[c2, 1]
    max_dim_length = torch.cat([h.unsqueeze(0), w.unsqueeze(0)]).max(dim=0).values.unsqueeze(-1)
    left_corner = oriented_representation.centers - max_dim_length / 2.0
    right_corner = oriented_representation.centers + max_dim_length / 2.0
    return left_corner.int(), right_corner.int()


def slice_patches(image, all_labels, left_corners, right_corners):
    # Slice the patches from the image using tensor indexing
    sliced_patches = []
    filtered_labels = []
    for ind, (curr_left_corner, curr_right_corner, label) in enumerate(zip(left_corners, right_corners, all_labels)):
        x_start, y_start = curr_left_corner[0], curr_left_corner[1]
        x_end, y_end = curr_right_corner[0], curr_right_corner[1]
        sliced_patch = image[:, y_start:y_end, x_start:x_end]
        if sliced_patch.shape[1] != 0 and sliced_patch.shape[2] != 0:
            sliced_patches.append(sliced_patch)
            filtered_labels.append(label)
    # print("Num of boxes is " + str(len(filtered_labels)))
    return sliced_patches, filtered_labels


def get_gt_bboxes_and_labels(gt_instances, image):
    bounding_square_left_corner, bounding_square_right_corner = convert_oriented_representation_to_bounding_square(
        gt_instances.bboxes, image.shape)
    bboxes, labels = slice_patches(image, gt_instances.labels,
                                   bounding_square_left_corner, bounding_square_right_corner)

    return bboxes, torch.tensor(labels)


def calculate_batch_features_and_labels(data_batch, features_model: Sequential):
    features_model.eval()
    with torch.no_grad():
        outputs = features_model(data_batch[0].squeeze(dim=1).to(device)).cpu().detach()
        outputs = outputs.flatten(1)
    labels = data_batch[1]
    # metadata = data_batch[2]
    return outputs, labels  # , metadata


def printing_data_statistics(labels, is_train):
    fg_number = torch.nonzero(labels != 0).size()[0]
    bg_number = torch.nonzero(labels == 0).size()[0]
    title = "train" if is_train else "test"
    print(title + " data contain " + str(fg_number) + " fg samples and " + str(bg_number) + " bg samples\n")


def get_outliers(scores, labels, dataset, output_dir, k, outliers_num=10):
    min_k_fg_scores_indices_in_dataset = torch.tensor([])
    # save high score bgs and low score fgs
    bg_scores = scores[torch.nonzero(labels == 0).squeeze(dim=1)]
    top_k_bg_scores, top_indices_bg_scores = torch.topk(bg_scores, k=outliers_num)
    top_k_bg_scores_indices_in_dataset = torch.where(top_k_bg_scores.unsqueeze(1) == scores.unsqueeze(0))[1]
    fg_scores = scores[torch.nonzero(labels > 0).squeeze(dim=1)]
    if fg_scores.size(0) > 0:
        min_k_fg_scores, min_indices_fg_scores = torch.topk(fg_scores, k=outliers_num, largest=False)
        min_k_fg_scores_indices_in_dataset = torch.where(scores.unsqueeze(0) == min_k_fg_scores.unsqueeze(1))[1]
    # save all outliers
    path = os.path.join(output_dir, f"outliers_k{str(k)}")
    os.makedirs(path, exist_ok=True)
    save_outliers(indices=top_k_bg_scores_indices_in_dataset, dataset=dataset, path=path, is_bg=True)
    save_outliers(indices=min_k_fg_scores_indices_in_dataset, dataset=dataset, path=path, is_bg=False)


def save_outliers(indices, dataset, path, is_bg):
    classes = dataset.classes
    for i in range(indices.size(0)):
        patch, label = dataset[indices[i]]
        title = f"{i+1} highest score bg, class {classes[label]}" if is_bg else \
            f"{i+1} lowest score fg, class {classes[label]}"
        saved_path = os.path.join(path, f"outlier_bg_{i+1}") if is_bg else os.path.join(path, f"outlier_fg_{i+1}")
        show_img(transform_to_imshow(patch), title=title, path=saved_path)


def cache_features_dictionary(features_model, dataset_cfg, target_dictionary_path: str, sampled_ratio:float):
    patches_dataset_cfg = dataset_cfg.get("patches_dataset")
    dataset_path = os.path.join(patches_dataset_cfg["path"], "subtrain", "images")
    from image_pyramid_patches_dataset import transform
    # patches_dataset = image_pyramid_patches_dataset(dataset_path)
    remove_empty_folders(dataset_path)
    patches_dataset = torchvision.datasets.DatasetFolder(dataset_path, extensions=(".png"), transform=transform,
                                                         loader=Image.open)
    data_loader = DataLoader(patches_dataset, batch_size=patches_dataset_cfg["batch_size"],
                             shuffle=patches_dataset_cfg["shuffle"], num_workers=patches_dataset_cfg["num_workers"])
    labels = []
    # preallocate memory
    sample_per_batch = int(data_loader.batch_size * sampled_ratio)
    features = torch.zeros((len(data_loader)*sample_per_batch, 2048)).cpu().detach()

    for idx, data_batch in tqdm(enumerate(data_loader)):
        images_patches_features, patches_labels = calculate_batch_features_and_labels(
            data_batch, features_model)
        # sample sampled_ratio out of the patches
        random_indices = torch.randperm(images_patches_features.size(0))[:int(sampled_ratio*images_patches_features.size(0))]
        sampled_features = torch.index_select(images_patches_features, 0, random_indices)
        sampled_labels = torch.index_select(patches_labels, 0, random_indices)
        # features.append(sampled_features.cpu().detach())
        features[idx*sample_per_batch:sample_per_batch*(idx+1), :]=sampled_features.cpu().detach()
        labels.append(sampled_labels)
    # features=torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    labels[torch.nonzero(labels > 0)] = 1
    # labels[torch.nonzero(labels == -1)] = 0
    data = {'features': features, 'labels': labels}
    torch.save(data, target_dictionary_path)


def calculate_scores_and_labels_for_test_dataset(dictionary: Dict, dataloader: DataLoader, features_model: Sequential,
                                                 k: int, output_dir: str, outliers_num: int):
    scores = []
    labels = []
    for idx, data_batch in tqdm(enumerate(dataloader)):
        images_patches_features, patches_labels = calculate_batch_features_and_labels(data_batch, features_model)
        batch_scores, batch_labels = calculate_scores_and_labels(images_patches_features, patches_labels,
                                                                 dictionary, k)
        scores.append(batch_scores)
        labels.append(batch_labels)

    scores = torch.concat(scores, dim=0)
    labels = torch.concat(labels, dim=0)
    get_outliers(scores=scores, labels=labels, dataset=dataloader.dataset, output_dir=output_dir, k=k,
                 outliers_num=outliers_num)
    labels[torch.nonzero(labels > 0)] = 1
    # save score and labels per k
    with open(os.path.join(output_dir, f'k_{str(k)}_scores_and_labels.json'), 'w') as f:
        k_dict = {}
        k_dict["scores"] = scores.tolist()
        k_dict["labels"] = labels.tolist()
        json.dump(k_dict, f, indent=4)
    return scores, labels


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="The relative path to the cfg file")
    parser.add_argument("-o", "--output_dir", help="Statistics output dir")
    parser.add_argument("-sr", "--sampled_ratio", default=0.5, help="Sampled ratio for the features dictionary")
    parser.add_argument("-k", "--k_values", type=int, nargs='+', help="Nearest Neighbours count.")
    parser.add_argument("-use-cached", "--use_cached", action='store_true',
                        help="If flagged, use the cached feature dict. "
                             "Otherwise, recalculate it every time you run the script.")

    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    model = models.resnet50(pretrained=True)
    model.eval()
    model = model.to(device)
    features_model = torch.nn.Sequential(*list(model.children())[:-1])
    features_model.eval()

    # Create features for the training stage in a case it does not exist
    dictionary_path = os.path.join(args.output_dir, 'features_dict_new_pyramid.pt')  # 'patches_dataset_dictionary.pt')
    if not args.use_cached:
        print('Recaching features dictionary...')
        cache_features_dictionary(features_model, cfg, target_dictionary_path=dictionary_path, sampled_ratio=args.sampled_ratio)
    # Test stage
    data = torch.load(dictionary_path)
    printing_data_statistics(data['labels'], is_train=True)

    patches_dataset_cfg = cfg.get("patches_dataset")
    dataset_path = os.path.join(patches_dataset_cfg["path"], "subval", "images")
    from image_pyramid_patches_dataset import transform
    # patches_dataset = image_pyramid_patches_dataset(dataset_path)
    remove_empty_folders(dataset_path)
    patches_dataset = torchvision.datasets.DatasetFolder(dataset_path, extensions=(".png"), transform=transform,
                                                         loader=Image.open)
    data_loader = DataLoader(patches_dataset, batch_size=patches_dataset_cfg["batch_size"],
                             shuffle=patches_dataset_cfg["shuffle"], num_workers=patches_dataset_cfg["num_workers"])
    k_to_auc = {}
    k_to_ap = {}

    for k_value in args.k_values:
        print(f"Evaluating scores and labels for k={k_value}...")
        scores, labels = calculate_scores_and_labels_for_test_dataset(dictionary=data['features'],
                                                                      dataloader=data_loader,
                                                                      features_model=features_model,
                                                                      k=int(k_value),
                                                                      output_dir=args.output_dir,
                                                                      outliers_num=20)

        printing_data_statistics(labels, is_train=False)
        plot_scores_histograms(scores.tolist(), labels.tolist(), k_value, args.output_dir)
        ap = plot_precision_recall_curve(labels, scores, k_value, args.output_dir)
        auc = plot_roc_curve(labels, scores, k_value, args.output_dir)
        k_to_auc[k_value] = auc
        k_to_ap[k_value] = ap

    with open(os.path.join(args.output_dir, 'k_to_auc_supervised.json'), 'w') as f:
        json.dump(k_to_auc, f, indent=4)

    with open(os.path.join(args.output_dir, 'k_to_ap_supervised.json'), 'w') as f:
        json.dump(k_to_ap, f, indent=4)


if __name__ == '__main__':
    main()
