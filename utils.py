import json
import pickle
import shutil
import logging
import os
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


def plot_outliers_images():
    from matplotlib import pyplot as plt

    # Path to the folder containing the images
    folder_path = "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/test/OOD/weights_in_loss_and_sampling_bg_in_val_and_train_datasets/outliers/50_highest_scores_patches"
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    # Define the number of images to display per row and column
    num_images_per_row = 10
    num_images_per_col = 5

    # Create a new figure
    fig, axs = plt.subplots(num_images_per_col, num_images_per_row, tight_layout=True)

    # Loop through the images and display them on the plot
    for i, file in enumerate(image_files):
        # Calculate the position of the image in the grid
        row = i // num_images_per_row
        col = i % num_images_per_row

        # Load the image using PIL
        img = Image.open(os.path.join(folder_path, file))

        # Display the image on the plot
        axs[row, col].imshow(img)

        axs[row, col].axis('off')

        # Set the title of the image as its name
        axs[row, col].set_title(file[:file.find("_")], fontsize=7)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=1)
    # Show the plot
    plt.savefig(os.path.dirname(folder_path) + "/50_highest_scores_patches.jpg")


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
                                   data_path):
    with open(hashmap_locations_and_anomaly_scores_test_file, 'r') as f:
        hashmap_locations_and_anomaly_scores_test = json.load(f)
    all_cache = threshold_scores(all_cache, data_path)
    anomaly_scores = []
    anomaly_scores_conv = []
    ood_scores = []
    labels = []
    for batch_index, batch in enumerate(test_dataloader):
        for path in batch['path']:
            img_id = os.path.basename(path).split('.')[0]
            if img_id in all_cache:
                curr_file_data = hashmap_locations_and_anomaly_scores_test[img_id]
                anomaly_scores.append(curr_file_data['anomaly_score'])
                anomaly_scores_conv.append(curr_file_data['anomaly_score_conv'])
                labels.append(all_cache[img_id]['label'])
                ood_scores.append(all_cache[img_id]['score'])

    return torch.tensor(ood_scores), torch.tensor(labels), anomaly_scores, anomaly_scores_conv

def threshold_scores(all_cache, data_path, dynamic_threshold=False):
    all_cache = dict(
        sorted(all_cache.items(), key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', x[0])]))
    images_names = np.unique([key[:key.find("__")] for key in all_cache.keys()])
    keys_to_delete = []
    for img in images_names:
        filter_thresh=250
        if dynamic_threshold:
            path = os.path.join(data_path, "DOTAV2_ss/test/images/")
            filter_thresh = len([f for f in os.listdir(path) if f.startswith(img)])*100
        scores = [all_cache[key]['score'] for key in all_cache.keys() if key[:key.find("__")] == img]
        sorted_scores = sorted(scores)[:filter_thresh]
        threshold = sorted_scores[-1]
        for key in all_cache.keys():
            if key[:key.find("__")] == img and all_cache[key]['score'] > threshold:
                keys_to_delete.append(key)
    for key in keys_to_delete:
        del all_cache[key]
    return all_cache

if __name__ == '__main__':
    plot_outliers_images()
