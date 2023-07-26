import json
from argparse import ArgumentParser
from torchvision.utils import save_image
from Anomaly_detection_BG_FG import create_dataloader, create_gaussian_pyramid, split_images_into_patches, \
    compute_patches_indices_per_scale
from mmengine.config import Config
import math
import os


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


def create_image_pyramid_patches_dataset(args):
    cfg = Config.fromfile(args.config)
    image_size = cfg["train_pipeline"][3]['scale']
    dataset_type = 'subval'
    os.makedirs(os.path.join(args.dataset_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_dir, "metadata"), exist_ok=True)
    data_loader = create_dataloader(cfg, mode=dataset_type)
    patches_indices_list = calc_patches_indices(args.pyramid_levels, args.patch_size, args.scale_factor,
                                                args.patch_stride, image_size)
    for idx, data_batch in enumerate(data_loader):
        pyramid_batch = create_gaussian_pyramid(data_batch['inputs'], pyramid_levels=args.pyramid_levels,
                                                scale_factor=args.scale_factor)
        curr_image_id = data_batch["data_samples"][0].img_id
        # extract patches
        patches_metadata = {}
        patch_num = 0
        for level_ind, pyramid_level_data in enumerate(pyramid_batch):
            patches = split_images_into_patches(pyramid_level_data,
                                                args.patch_size, args.patch_stride)
            for patch_ind in range(patches.shape[1]):
                patch = patches[0, patch_ind]
                curr_patch_metadata = {}
                patch_name = curr_image_id + "_" + str(patch_num)
                curr_patch_metadata["origin_image"] = curr_image_id
                curr_patch_metadata["scale"] = args.scale_factor ** level_ind
                curr_patch_metadata["indices"] = patches_indices_list[level_ind][patch_ind].tolist()
                patches_metadata[patch_name] = curr_patch_metadata
                patch_save_path = os.path.join(args.dataset_dir, "images", patch_name + ".png")
                save_image(patch.unsqueeze(0), patch_save_path, normalize=True)
                patch_num += 1
        with open(os.path.join(args.dataset_dir, "metadata", curr_image_id + ".json"), 'w') as file:
            json.dump(patches_metadata, file)


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="The relative path to the cfg file")
    parser.add_argument("-d", "--dataset_dir", help="The saved model path")
    parser.add_argument("-l", "--pyramid_levels", default=5, help="Number of pyramid levels")
    parser.add_argument("-sf", "--scale_factor", default=0.6, help="Scale factor between pyramid levels")
    parser.add_argument("-po", "--patch_stride", default=0.1, help="Stride between patches in %")
    parser.add_argument("-ps", "--patch_size", default=50, help="The patch size")
    args = parser.parse_args()
    create_image_pyramid_patches_dataset(args)


if __name__ == '__main__':
    main()
