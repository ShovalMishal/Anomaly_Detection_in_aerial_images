import json
import pickle
import shutil
import logging
import os
import sys

import torch
from matplotlib import pyplot as plt
from plotly.figure_factory import np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from ImagePyramidPatchesDataset import ImagePyramidPatchesDataset
from mmengine.runner import Runner
import copy
from PIL import Image
import re
import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_logger(path):
    if os.path.exists(path):
        os.remove(path)
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def remove_empty_folders(path, logger):
    for root, dirs, files in os.walk(path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # Check if the folder is empty
                logger.info(f"Removing empty folder: {dir_path}")
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


def create_dataloader(dataloader_cfg):
    dataloader_cfg = copy.deepcopy(dataloader_cfg)
    dataloader_cfg["dataset"]["_scope_"] = "mmrotate"
    data_loader = Runner.build_dataloader(dataloader_cfg)
    return data_loader


def create_patches_dataset(type, dataset_cfg, transform, output_dir, logger, is_valid_file_use=False, ood_remove=False):
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
    remove_empty_folders(dataset_images_path, logger)
    if ood_remove:
        remove_ood_folders(dataset_images_path, dataset_cfg.ood_classes)
    save_samples_filename_splitted = dataset_cfg.inclusion_file_name.split("_", 1)
    inclusion_file_path = os.path.join(output_dir,
                                       f"train/anomaly_detection_result/{save_samples_filename_splitted[0]}_{type}_{save_samples_filename_splitted[1]}")
    is_valid_file = None
    if is_valid_file_use:
        is_valid_file = is_valid_file_wrapper(inclusion_file_path)
    patches_dataset = ImagePyramidPatchesDataset(root_dir=dataset_images_path, inclusion_file_path=inclusion_file_path,
                                                 dataset_type=type, output_dir=output_dir, logger=logger,
                                                 transform=transform,
                                                 is_valid_file=is_valid_file, ood_classes_names=dataset_cfg.ood_classes)
    return patches_dataset


def calculate_confusion_matrix(dataloader, model, logger, dataset_name="", path=""):
    save_path = os.path.join(path, f"confusion_matrix_{dataset_name}_dataset.json")
    all_preds, all_labels = None, None
    if not os.path.exists(save_path):
        all_preds, all_labels = eval_model(dataloader, model)
        # Generate the confusion matrix
    create_and_save_confusion_matrix(path=path, dataset_name=dataset_name, logger=logger, all_preds=all_preds,
                                     all_labels=all_labels)


def eval_model(dataloader, model, cache_dir=""):
    cache_dir = os.path.join(cache_dir, "test_results")
    os.makedirs(cache_dir, exist_ok=True)
    all_preds = []
    all_labels = []
    model = model.to(device)
    model.eval()
    next_cache_index = 0
    curr_cache_index = 0

    for batch_ind, batch in tqdm(enumerate(dataloader)):
        curr_cache_index = batch_ind // 1000
        file_related_path = os.path.join(cache_dir, f'all_preds_cache_{curr_cache_index}.pkl')
        if os.path.exists(file_related_path):
            next_cache_index = curr_cache_index + 1
            del batch
            continue

        if curr_cache_index != next_cache_index:
            print(next_cache_index)
            with open(os.path.join(cache_dir, f'all_preds_cache_{next_cache_index}.pkl'), 'wb') as f:
                pickle.dump(all_preds, f)
            with open(os.path.join(cache_dir, f'all_labels_cache_{next_cache_index}.pkl'), 'wb') as f:
                pickle.dump(all_labels, f)
            all_labels = []
            all_preds = []
            next_cache_index = curr_cache_index
        images = batch['pixel_values']
        labels = batch['labels']
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
        try:
            pred_labels = torch.argmax(pred["logits"].data.cpu().detach(), dim=1)
        except:
            pred_labels = torch.argmax(pred.data.cpu().detach(), dim=1)
        all_preds.append(pred_labels)
        all_labels.append(labels.cpu())

        del images, labels, pred
        torch.cuda.empty_cache()

    file_related_path = os.path.join(cache_dir, f'all_preds_cache_{curr_cache_index}.pkl')
    if not os.path.exists(file_related_path):
        with open(os.path.join(cache_dir, f'all_preds_cache_{curr_cache_index}.pkl'), 'wb') as f:
            pickle.dump(all_preds, f)
        with open(os.path.join(cache_dir, f'all_labels_cache_{curr_cache_index}.pkl'), 'wb') as f:
            pickle.dump(all_labels, f)
    all_labels_caches = [x for x in os.listdir(cache_dir) if x.startswith('all_labels_cache_')]
    all_labels_caches.sort(key=lambda x: int(x.split('all_labels_cache_')[-1].split('.pkl')[0]))
    all_labels = []
    for cache_name in all_labels_caches:
        with open(os.path.join(cache_dir, cache_name), 'rb') as f:
            all_labels.append(pickle.load(f))
    all_labels = sum(all_labels, [])
    all_preds_caches = [x for x in os.listdir(cache_dir) if x.startswith('all_preds_cache_')]
    all_preds_caches.sort(key=lambda x: int(x.split('all_preds_cache_')[-1].split('.pkl')[0]))
    all_preds = []
    for cache_name in all_preds_caches:
        with open(os.path.join(cache_dir, cache_name), 'rb') as f:
            all_preds.append(pickle.load(f))
    all_preds = sum(all_preds, [])

    all_preds_final, all_labels_final = torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)
    assert all_preds_final.shape[0] == all_labels_final.shape[0], "lables and preds lengths are not equal"
    return torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)


def create_and_save_confusion_matrix(path, dataset_name, logger, all_labels=None, all_preds=None):
    save_path = os.path.join(path, f"confusion_matrix_{dataset_name}_dataset.json")
    if not os.path.exists(save_path):
        res_confusion_matrix = confusion_matrix(all_labels, all_preds)
        with open(save_path, 'w') as f:
            json.dump(res_confusion_matrix.tolist(), f, indent=4)
    with open(save_path, 'r') as f:
        res_confusion_matrix = json.load(f)
    logger.info(f"Confusion Matrix for {dataset_name} dataset\n")
    logger.info(res_confusion_matrix)


def retrieve_scores_for_test_dataset(test_dataloader, hashmap_locations_and_anomaly_scores_test_file, all_cache):
    with open(hashmap_locations_and_anomaly_scores_test_file, 'r') as f:
        hashmap_locations_and_anomaly_scores_test = json.load(f)

    anomaly_scores = []
    anomaly_scores_conv = []
    ood_scores = []
    labels = []
    for batch_index, batch in enumerate(test_dataloader):
        for path in batch['path']:
            img_id = os.path.basename(path).split('.')[0]
            curr_file_data = hashmap_locations_and_anomaly_scores_test[img_id]
            anomaly_scores.append(curr_file_data['anomaly_score'])
            anomaly_scores_conv.append(curr_file_data['anomaly_score_conv'])
            labels.append(all_cache[img_id]['label'])
            ood_scores.append(all_cache[img_id]['score'])

    return torch.tensor(ood_scores), torch.tensor(labels), anomaly_scores, anomaly_scores_conv

def threshold_and_retrieve_samples(test_dataloader, hashmap_locations_and_anomaly_scores_test_file, all_cache,
                                   data_path, cache_path, patches_per_image):
    if not os.path.exists(cache_path):
        with open(hashmap_locations_and_anomaly_scores_test_file, 'r') as f:
            hashmap_locations_and_anomaly_scores_test = json.load(f)
        all_cache = threshold_scores(all_cache, data_path, patches_per_image)
        anomaly_scores = []
        anomaly_scores_conv = []
        ood_scores = []
        labels = []
        original_inds = []
        for batch_index, batch in enumerate(test_dataloader):
            for path_ind, path in enumerate(batch['path']):
                img_id = os.path.basename(path).split('.')[0]
                if img_id in all_cache:
                    curr_file_data = hashmap_locations_and_anomaly_scores_test[img_id]
                    anomaly_scores.append(curr_file_data['anomaly_score'])
                    anomaly_scores_conv.append(curr_file_data['anomaly_score_conv'])
                    labels.append(all_cache[img_id]['label'])
                    ood_scores.append(all_cache[img_id]['score'])
                    original_inds.append(batch_index*len(batch['path'])+path_ind)
        filtered_cache={}
        filtered_cache["ood_scores"]=ood_scores
        filtered_cache["labels"]=labels
        filtered_cache["anomaly_scores"]=anomaly_scores
        filtered_cache["anomaly_scores_conv"]=anomaly_scores_conv
        filtered_cache["original_inds"]=original_inds
        with open(cache_path, 'w') as f:
            json.dump(filtered_cache, f, indent=4)
    else:
        with open(cache_path, 'r') as file:
            filtered_cache = json.load(file)
        ood_scores, labels, anomaly_scores, anomaly_scores_conv, original_inds = filtered_cache["ood_scores"], filtered_cache["labels"], filtered_cache["anomaly_scores"], filtered_cache["anomaly_scores_conv"], filtered_cache["original_inds"]
    return torch.tensor(ood_scores), torch.tensor(labels), anomaly_scores, anomaly_scores_conv, torch.tensor(original_inds)

def threshold_scores(all_cache, data_path, patches_per_image, dynamic_threshold=False):
    all_cache = dict(
        sorted(all_cache.items(), key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', x[0])]))
    images_names = np.unique([key[:key.find("__")] for key in all_cache.keys()])
    keys_to_delete = []
    for img_ind, img  in tqdm(enumerate(images_names)):
        if dynamic_threshold:
            path = os.path.join(data_path, "DOTAV2_ss/test/images/")
            patches_per_image = len([f for f in os.listdir(path) if f.startswith(img)])*100
        scores = [all_cache[key]['score'] for key in all_cache.keys() if key[:key.find("__")] == img]
        sorted_scores = sorted(scores)[:patches_per_image]
        threshold = sorted_scores[-1]
        for key in all_cache.keys():
            if key[:key.find("__")] == img and all_cache[key]['score'] > threshold:
                keys_to_delete.append(key)
    for key in keys_to_delete:
        del all_cache[key]
    return all_cache


class ResizeLargestAndPad:
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        w, h = image.size
        # Determine the scaling factor to make the largest side equal to size
        if w > h:
            new_w, new_h = self.size, int(h * self.size / w)
        else:
            new_w, new_h = int(w * self.size / h), self.size

        # Resize the image with the aspect ratio preserved
        resized_image = transforms.functional.resize(image, (new_h, new_w))

        # Calculate the padding to make the image square
        padding_left = (self.size - new_w) // 2
        padding_top = (self.size - new_h) // 2
        padding_right = self.size - new_w - padding_left
        padding_bottom = self.size - new_h - padding_top

        # Apply padding
        padded_image = transforms.functional.pad(resized_image,
                                                 (padding_left, padding_top, padding_right, padding_bottom), fill=0)

        return padded_image

def rank_TT_1_acord_scores(all_labels, all_scores, ood_labels, labels_to_classes_names, logger, OOD=False, plot=False):
    mode = "OOD" if OOD else "AD"
    all_scores = torch.tensor(all_scores)
    all_labels = torch.tensor(all_labels)
    sorted_all_scores, sorted_all_scores_indices = torch.sort(all_scores)

    tt1 = {}
    ranks_dict = {}
    for OOD_label in ood_labels:
        if OOD_label not in all_labels:
            continue
        curr_label_scores = all_scores[all_labels == OOD_label]
        curr_label_sorted_scores, curr_label_anomaly_scores_indices = torch.sort(
            curr_label_scores)
        curr_label_ranks_in_all_scores = len(
            sorted_all_scores) - 1 - torch.searchsorted(sorted_all_scores,
                                                                curr_label_sorted_scores)
        sorted_curr_label_ranks_in_all_scores, _ = torch.sort(curr_label_ranks_in_all_scores)
        tt1[labels_to_classes_names[OOD_label]] = \
            sorted_curr_label_ranks_in_all_scores[0]
        logger.info(
            f"OOD label {labels_to_classes_names[OOD_label]} first rank in {mode}"
            f" scores: {tt1[labels_to_classes_names[OOD_label]]}")
        ranks_dict[labels_to_classes_names[OOD_label]] = list(
            sorted_curr_label_ranks_in_all_scores.numpy().astype(np.float64))
        if plot:
            plt.plot(list(range(len(curr_label_ranks_in_all_scores))),
                     sorted_curr_label_ranks_in_all_scores, label=labels_to_classes_names[OOD_label])

    return tt1, ranks_dict

def create_ood_id_dataset(src_root: str, target_root: str, ood_classes: list):
    # create src root and target root directories
    os.makedirs(target_root, exist_ok=True)
    os.makedirs(os.path.join(target_root, 'id_dataset'), exist_ok=True)
    os.makedirs(os.path.join(target_root, 'ood_dataset'), exist_ok=True)
    for _class in os.listdir(src_root):
        if _class not in ood_classes:
            if not os.path.exists(os.path.join(target_root, 'id_dataset', _class)):
                os.symlink(os.path.join(src_root, _class), os.path.join(target_root, 'id_dataset', _class),
                           target_is_directory=True)

    for _class in ood_classes:
        if not os.path.exists(os.path.join(target_root, 'ood_dataset', _class)) and os.path.exists(os.path.join(src_root, _class)):
            os.symlink(os.path.join(src_root, _class), os.path.join(target_root, 'ood_dataset', _class),
                       target_is_directory=True)

