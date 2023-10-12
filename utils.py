import json
import os
import shutil
import random
import os
import torch
from matplotlib import pyplot as plt
from plotly.figure_factory import np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from ImagePyramidPatchesDataset import ImagePyramidPatchesDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def remove_empty_folders(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # Check if the folder is empty
                print(f"Removing empty folder: {dir_path}")
                os.rmdir(dir_path)


def remove_ood_folders(dataset_path, ood_names):
    for name in ood_names:
        full_dir_path = os.path.join(dataset_path, name)
        if os.path.exists(full_dir_path):
            shutil.rmtree(full_dir_path)


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
        inclusion_set = set(line.strip() for line in f)

    def is_valid_file(filename):
        return filename in inclusion_set

    return is_valid_file


def create_patches_dataloader(type, dataset_cfg, batch_size, transform, output_dir, is_valid_file_use=False,
                              collate_fn=None):
    patches_dataset = create_patches_dataset(type, dataset_cfg, transform, output_dir, is_valid_file_use)
    data_loader = DataLoader(dataset=patches_dataset, batch_size=batch_size, shuffle=dataset_cfg["shuffle"],
                             num_workers=dataset_cfg["num_workers"], collate_fn=collate_fn)
    return data_loader


def create_patches_dataset(type, dataset_cfg, transform, output_dir, is_valid_file_use=False, ood_remove=False):
    dataset_path = os.path.join(dataset_cfg["path"], type)
    dataset_images_path = os.path.join(dataset_path, "images")
    # make sure there is a backup for subtrain and subval. and copy all ood classes folders to the dataset images path.
    if "val" in type or "train" in type:
        if not os.path.exists(dataset_path + " (copy)"):
            shutil.copytree(src=dataset_path, dst=dataset_path + " (copy)")
        for ood_class in dataset_cfg.ood_classes:
            if not os.path.exists(os.path.join(dataset_images_path, ood_class)):
                shutil.copytree(src=os.path.join(dataset_path + " (copy)", "images", ood_class),
                                dst=os.path.join(dataset_images_path, ood_class))
    remove_empty_folders(dataset_images_path)
    if ood_remove:
        remove_ood_folders(dataset_images_path, dataset_cfg.ood_classes)
    save_samples_filename_splitted = dataset_cfg.inclusion_file_name.split("_", 1)
    inclusion_file_path = os.path.join(output_dir,
                                       f"train/anomaly_detection_result/{save_samples_filename_splitted[0]}_{type}_{save_samples_filename_splitted[1]}")
    is_valid_file = None
    if is_valid_file_use:
        is_valid_file = is_valid_file_wrapper(inclusion_file_path)
    patches_dataset = ImagePyramidPatchesDataset(root_dir=dataset_images_path, inclusion_file_path=inclusion_file_path,
                                                 dataset_type=type, output_dir=output_dir, transform=transform,
                                                 is_valid_file=is_valid_file, ood_classes_names=dataset_cfg.ood_classes)
    return patches_dataset


def calculate_confusion_matrix(dataloader, model, dataset_name="", path=""):
    all_preds, all_labels = eval_model(dataloader, model)
    # Generate the confusion matrix
    create_and_save_confusion_matrix(all_labels, all_preds, dataset_name, path)


def eval_model(dataloader, model):
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader):
        images = batch['pixel_values']
        labels = batch['labels']
        pred = model(images.to(device))
        pred_labels = torch.argmax(pred["logits"].data.cpu().detach(), dim=1)
        all_preds.append(pred_labels)
        all_labels.append(labels)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_preds, all_labels


def create_and_save_confusion_matrix(all_labels, all_preds, dataset_name, path):
    save_path = os.path.join(path, f"confusion_matrix_{dataset_name}_dataset.json")
    if not os.path.exists(save_path):
        res_confusion_matrix = confusion_matrix(all_labels, all_preds)
        with open(save_path, 'w') as f:
            json.dump(res_confusion_matrix.tolist(), f, indent=4)
    else:
        with open(save_path, 'r') as f:
            res_confusion_matrix = json.load(f)
    print(f"Confusion Matrix for {dataset_name} dataset\n")
    print(res_confusion_matrix)

