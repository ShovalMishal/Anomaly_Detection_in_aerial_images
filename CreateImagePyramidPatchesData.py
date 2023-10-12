import json
from argparse import ArgumentParser
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from mmengine.runner import Runner
import copy
from mmengine.config import Config
import math
import os
from mmdet.models.utils import unpack_gt_instances
from mmengine.structures import InstanceData
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.registry import TASK_UTILS
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection
import numpy as np
from statistics import get_unfound_fg_areas_n_aspect_ratio, plot_fg_statistics
import skimage
import DOTA_devkit.dota_utils as util
from utils import show_img


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

    return bboxes, torch.tensor(labels), (bounding_square_left_corner, bounding_square_right_corner)

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
    indices = torch.stack((indices_tl_w, indices_tl_h, indices_br_w, indices_br_h), dim=0)
    indices = indices.view(4, -1).T.float()
    indices *= 1 / scale_factor
    return indices.int()

def split_images_into_patches(images, patch_size, patch_stride: float):
    stride = int(patch_size * patch_stride)

    patches = images.unfold(2, patch_size, stride).unfold(3, patch_size, stride)

    # Reshape the patches tensor to the desired shape
    B, C, _, _, _, _ = patches.size()
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.contiguous().view(B, -1, C, patch_size, patch_size)

    return patches

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

def create_dataloader(cfg, mode="train"):
    dataloader_mode = mode + "_dataloader"
    test_dataloader = cfg.get(dataloader_mode)
    dataloader_cfg = copy.deepcopy(test_dataloader)
    dataloader_cfg["dataset"]["_scope_"] = "mmrotate"
    data_loader = Runner.build_dataloader(dataloader_cfg, seed=123456)
    return data_loader

def calc_patches_indices(pyramid_levels: int, patch_size: int, scale_factor: float, patch_stride: float, image_size):
    patches_indices_list = []
    curr_image_size = image_size
    for i in range(pyramid_levels):
        scale = scale_factor ** i
        curr_scale_patches_indices = compute_patches_indices_per_scale(curr_image_size, patch_stride,
                                                                       patch_size,
                                                                       scale_factor=scale)
        curr_image_size = (math.ceil(scale_factor * curr_image_size[0]), math.ceil(scale_factor * curr_image_size[1]))
        patches_indices_list.append(curr_scale_patches_indices)
    return patches_indices_list


def extract_and_save_fgs(gt_instances, image, curr_image_id, scale_factor, dataset_type, dataset_dir, all_labels):
    patch_num = 0
    foreground_patches, foreground_labels, fg_patches_indices = get_gt_bboxes_and_labels(gt_instances, image)
    fg_patches_metadata = {}
    for patch_ind, patch in enumerate(foreground_patches):
        level_ind = 0
        curr_patch_metadata, patch_name = save_patch(patch, patch_num, curr_image_id, scale_factor,
                                                     fg_patches_indices[0][patch_ind].tolist() + \
                                                     fg_patches_indices[1][patch_ind].tolist(),
                                                     level_ind, dataset_type,
                                                     foreground_labels[patch_ind].item(),
                                                     dataset_dir, all_labels)
        fg_patches_metadata[patch_name] = curr_patch_metadata
        patch_num += 1
    return patch_num, fg_patches_metadata


def save_patch(patch, patch_num, curr_image_id, scale_factor, indices_list,
               level_ind, dataset_type, label_num, dataset_dir, all_labels):
    curr_patch_metadata = {}
    patch_name = curr_image_id + "_" + str(patch_num)
    label = all_labels[label_num] if label_num != -1 else "background"
    patch_save_path = os.path.join(dataset_dir, dataset_type, "images", label, patch_name + ".png")
    curr_patch_metadata["origin_image"] = curr_image_id
    curr_patch_metadata["scale"] = scale_factor ** level_ind
    curr_patch_metadata["indices"] = indices_list
    curr_patch_metadata["path"] = patch_save_path
    save_image(torch.flip(patch.to(torch.float32), dims=[0]).unsqueeze(0), patch_save_path, normalize=True)
    return curr_patch_metadata, patch_name


def create_image_pyramid_patches_dataset(args):
    cfg = Config.fromfile(args.config)
    bbox_assigner = TASK_UTILS.build(cfg["patches_assigner"])
    image_size = cfg["train_pipeline"][3]['scale']
    original_dataset_path = cfg[args.dataset_type + "_dataloader"]["dataset"]["data_root"]
    original_dataset_anns_path = os.path.join(original_dataset_path,
                                              cfg[args.dataset_type + "_dataloader"]["dataset"]["ann_file"])
    os.makedirs(os.path.join(args.dataset_dir, args.dataset_type, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_dir, args.dataset_type, "metadata"), exist_ok=True)
    data_loader = create_dataloader(cfg, mode=args.dataset_type)
    all_labels = data_loader.dataset.METAINFO['classes']
    # create class label folders
    os.makedirs(os.path.join(args.dataset_dir, args.dataset_type, "images", "background"), exist_ok=True)
    for label in all_labels:
        os.makedirs(os.path.join(args.dataset_dir, args.dataset_type, "images", label), exist_ok=True)
    patches_indices_list = calc_patches_indices(args.pyramid_levels, args.patch_size, args.scale_factor,
                                                args.patch_stride, image_size)
    all_unfound_fg_areas, all_unfound_fg_aspect_ratio, all_found_fg_areas, all_found_fg_aspect_ratio = \
        [], [], [], []
    assert data_loader.batch_size == 1, ("In this stage the dataloader should have batch with size 1 due to "
                                         "memory constraints!")
    patches_dataset_len = 0
    for idx, data_batch in tqdm(enumerate(data_loader)):
        # show image
        # objects = util.parse_dota_poly(os.path.join(original_dataset_anns_path, data_batch['data_samples'][0].img_id+".txt"))
        # showAnns(objects, np.transpose(data_batch['inputs'][0].numpy(), (1, 2, 0)))
        pyramid_batch = create_gaussian_pyramid(data_batch['inputs'], pyramid_levels=args.pyramid_levels,
                                                scale_factor=args.scale_factor)

        curr_image_id = data_batch["data_samples"][0].img_id
        # extract patches
        patches_metadata = {}
        patch_num = 0
        gt_instances = unpack_gt_instances(data_batch['data_samples'])
        batch_gt_instances, batch_gt_instances_ignore, _ = gt_instances
        if "val" in args.dataset_type or "test" in args.dataset_type:
            # save all fg
            patch_num, fg_patches_metadata = extract_and_save_fgs(batch_gt_instances[0], data_batch['inputs'][0],
                                                                  curr_image_id, args.scale_factor, args.dataset_type,
                                                                  args.dataset_dir, all_labels)
            patches_metadata.update(fg_patches_metadata)
        total_gt_ind = []
        for level_ind, pyramid_level_data in enumerate(pyramid_batch):
            patches = split_images_into_patches(pyramid_level_data,
                                                args.patch_size, args.patch_stride)
            patches_horizontal_boxes = HorizontalBoxes(patches_indices_list[level_ind])
            formatted_patches_indices = InstanceData(priors=patches_horizontal_boxes)
            assign_result = bbox_assigner.assign(
                formatted_patches_indices, batch_gt_instances[0],
                batch_gt_instances_ignore[0])
            assigned_labels = assign_result.labels
            total_gt_ind.append(assign_result.gt_inds)
            if "val" in args.dataset_type or "test" in args.dataset_type:
                # filter patches to obtain only bgs
                indices_of_bbs_with_iou_bellow_threshold = torch.squeeze(torch.nonzero(assign_result.gt_inds == 0))
                assigned_labels = assign_result.labels[indices_of_bbs_with_iou_bellow_threshold]
                patches = patches[:, indices_of_bbs_with_iou_bellow_threshold]
            for patch_ind in range(patches.shape[1]):
                patch = patches[0, patch_ind]
                # if(assigned_labels[patch_ind] != -1):
                #     show_img(patch, all_labels[assigned_labels[patch_ind]])
                curr_patch_metadata, patch_name = save_patch(patch, patch_num, curr_image_id, args.scale_factor,
                                                             patches_indices_list[level_ind][patch_ind].tolist(),
                                                             level_ind, args.dataset_type,
                                                             assigned_labels[patch_ind].item(),
                                                             args.dataset_dir, all_labels)
                patches_metadata[patch_name] = curr_patch_metadata
                patch_num += 1
        patches_dataset_len+=patch_num
        with open(os.path.join(args.dataset_dir, args.dataset_type, "metadata", curr_image_id + ".json"), 'w') as file:
            json.dump(patches_metadata, file, indent=4)

        total_gt_ind = torch.cat(total_gt_ind)
        if "train" in args.dataset_type:
            fg_gt_bboxes = batch_gt_instances[0].bboxes
            founded_fg_indices = torch.unique(total_gt_ind[torch.nonzero(total_gt_ind > 0)]) - 1
            unfounded_areas, unfounded_aspect_ratio, founded_areas, founded_aspect_ratio = \
                get_unfound_fg_areas_n_aspect_ratio(fg_gt_bboxes, founded_fg_indices)
            all_unfound_fg_areas.append(unfounded_areas)
            all_unfound_fg_aspect_ratio.append(unfounded_aspect_ratio)
            all_found_fg_areas.append(founded_areas)
            all_found_fg_aspect_ratio.append(founded_aspect_ratio)
            number_of_all_filtered_fg = torch.nonzero(unfounded_areas > 1000).size(0) + founded_areas.size(0)
            print(f"({idx}) The number of patches match to unique fg bboxes is " +
                  str(founded_fg_indices.shape[0])
                  + " out of " + str(number_of_all_filtered_fg))
    print(f"For {args.dataset_type} dataset {patches_dataset_len} patches were created!")
    if "train" in args.dataset_type:
        plot_fg_statistics(torch.cat(all_unfound_fg_areas), torch.cat(all_unfound_fg_aspect_ratio),
                           is_unfound=True)
        plot_fg_statistics(torch.cat(all_found_fg_areas), torch.cat(all_found_fg_aspect_ratio),
                           is_unfound=False)


def showAnns(objects, img):
    """
    :param objects: objects to show
    :param img: img to show
    :return:
    """
    rgb_image = np.flip(img, axis=-1)
    plt.imshow(rgb_image)
    plt.axis('off')

    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    circles = []
    r = 5
    for obj in objects:
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        poly = obj['poly']
        polygons.append(Polygon(poly))
        color.append(c)
        point = poly[0]
        circle = Circle((point[0], point[1]), r)
        circles.append(circle)
    p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)
    p = PatchCollection(circles, facecolors='red')
    ax.add_collection(p)
    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="The relative path to the cfg file")
    parser.add_argument("-d", "--dataset_dir", help="The saved dataset path")
    parser.add_argument("-l", "--pyramid_levels", default=6, help="Number of pyramid levels")
    parser.add_argument("-sf", "--scale_factor", default=0.7, help="Scale factor between pyramid levels")
    parser.add_argument("-po", "--patch_stride", default=0.1, help="Stride between patches in %")
    parser.add_argument("-ps", "--patch_size", default=50, help="The patch size")
    parser.add_argument("-dt", "--dataset_type", default="subval", help="The dataset type (train or test)")
    args = parser.parse_args()
    create_image_pyramid_patches_dataset(args)


if __name__ == '__main__':
    main()
