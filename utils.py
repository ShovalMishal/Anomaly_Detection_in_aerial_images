import os

import torch
from matplotlib import pyplot as plt
from plotly.figure_factory import np
from torch.utils.data import DataLoader

from ImagePyramidPatchesDataset import ImagePyramidPatchesDataset


def remove_empty_folders(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # Check if the folder is empty
                print(f"Removing empty folder: {dir_path}")
                os.rmdir(dir_path)


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

def is_valid_file_wrapper(path_to_exclusion_file):
    with open(path_to_exclusion_file, 'r') as f:
        data = f.read()

    def is_valid_file(filename):
        return filename in data.splitlines()

    return is_valid_file

def create_patches_dataloader(type, dataset_cfg, transform, output_dir, is_valid_file_use=False):
    dataset_path = os.path.join(dataset_cfg["path"], type, "images")
    remove_empty_folders(dataset_path)
    save_samples_filename_splitted = dataset_cfg.inclusion_file_name.split("_", 1)
    inclusion_file_path = os.path.join(output_dir,
                                            f"train/anomaly_detection_result/{save_samples_filename_splitted[0]}_{type}_{save_samples_filename_splitted[1]}")
    is_valid_file = None
    if is_valid_file_use:
        is_valid_file = is_valid_file_wrapper(inclusion_file_path)
    patches_dataset = ImagePyramidPatchesDataset(root_dir=dataset_path, inclusion_file_path=inclusion_file_path,
                                                 dataset_type=type, output_dir=output_dir, transform=transform,
                                                 is_valid_file=is_valid_file, ood_classes_names=dataset_cfg.ood_classes)
    data_loader = DataLoader(patches_dataset, batch_size=dataset_cfg["batch_size"],
                             shuffle=dataset_cfg["shuffle"], num_workers=dataset_cfg["num_workers"])
    return data_loader