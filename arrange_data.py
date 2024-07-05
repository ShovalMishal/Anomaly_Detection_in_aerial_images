import json
import os
import shutil
import sys
from collections import defaultdict
import statistics

import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from mmengine.config import Config
import argparse

from DOTA_devkit.DOTA import DOTA
from utils import create_dataloader
import random


def split_train_to_train_and_val_datasets(source_path):
    train_dataloader = dict(
        batch_size=1,
        num_workers=1,
        # num_workers=0,
        persistent_workers=True,
        # persistent_workers=False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type='DOTAv2Dataset',
            data_root=source_path,
            ann_file='labelTxt/',
            data_prefix=dict(img_path='images/'),
            ignore_ood_labels=False,
            pipeline=[
                dict(type='mmdet.LoadImageFromFile'),
                dict(
                    type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                dict(
                    type='ConvertBoxType',
                    # box_type_mapping=dict(gt_bboxes='hbox')),
                    box_type_mapping=dict(gt_bboxes='rbox')),
                dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
                dict(type='Normalize', mean=(0, 0, 0), std=(255.0, 255.0, 255.0), to_rgb=False),
                dict(type='Normalize', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
            ]))
    dest_path = os.path.join(os.path.dirname(source_path), "val")
    dest_image_folder = os.path.join(dest_path, "images")
    dest_labels_folder = os.path.join(dest_path, "labelTxt")
    os.makedirs(dest_image_folder, exist_ok=True)
    os.makedirs(dest_labels_folder, exist_ok=True)
    source_image_folder = os.path.join(source_path, "images")
    source_labels_folder = os.path.join(source_path, "labelTxt")
    train_dataloader = create_dataloader(train_dataloader)
    train_labels_counter = count_labels_representation_in_dataset(train_dataloader.dataset)
    desired_counter = {key:value//10 for key,value in train_labels_counter.items()}
    images_to_copy = []
    labels_to_classes_names = train_dataloader.dataset.METAINFO['classes']
    i=0
    while len(desired_counter) > 0:
        to_copy = False
        curr_labels = list(train_dataloader.dataset[i]['data_samples'].gt_instances.labels.numpy())
        counter = Counter(curr_labels)
        counter_labels = {labels_to_classes_names[key]:value for key,value in counter.items()}
        for label_name, label_count in counter_labels.items():
            if label_name in desired_counter and desired_counter[label_name]>0:
                to_copy = True
                desired_counter[label_name] -= counter_labels[label_name]
                if desired_counter[label_name] <= 0:
                    del desired_counter[label_name]
        if to_copy:
            images_to_copy.append(train_dataloader.dataset[i]['data_samples'].img_id)
        i+=1

    images_to_copy = list(set(images_to_copy))
    for img_name in images_to_copy:
        shutil.move(os.path.join(source_image_folder, img_name + '.png'), dest_image_folder)
        shutil.move(os.path.join(source_labels_folder, img_name + '.txt'), dest_labels_folder)

    print(f"{len(images_to_copy)} images were copied to val dataset")


def copy_data_and_sample_from_val_background():
    source_data = "/storage/shoval/Anomaly_Detection_in_aerial_images/results/train/anomaly_detection_result/extracted_bboxes_data"
    dest_data = "/storage/shoval/Anomaly_Detection_in_aerial_images/results/train/anomaly_detection_result/sampled_extracted_bboxes_data"
    if os.path.exists(dest_data):
        shutil.rmtree(dest_data)
    shutil.copytree(os.path.join(source_data, "val"), os.path.join(dest_data, "val"))
    shutil.copytree(os.path.join(source_data, "train"), os.path.join(dest_data, "train"))
    shutil.copytree(os.path.join(source_data, "test"), os.path.join(dest_data, "test"))

    # sample val bg
    val_background_dir = os.path.join(dest_data, 'val/background')
    val_background_files = os.listdir(val_background_dir)
    num_files_to_sample = max(1, len(val_background_files) // 100)
    print(f"{num_files_to_sample} files were sampled from val ds...")
    sampled_files = random.sample(val_background_files, num_files_to_sample)
    for file_name in val_background_files:
        if file_name not in sampled_files:
            file_path = os.path.join(val_background_dir, file_name)
            os.remove(file_path)

    # sample train bg
    train_background_dir = os.path.join(dest_data, 'train/background')
    train_background_files = os.listdir(train_background_dir)
    num_files_to_sample = max(1, len(train_background_files) // 10)
    print(f"{num_files_to_sample} files were sampled from train ds...")
    sampled_files = random.sample(train_background_files, num_files_to_sample)
    for file_name in train_background_files:
        if file_name not in sampled_files:
            file_path = os.path.join(train_background_dir, file_name)
            os.remove(file_path)



def calculate_dataset_representation(train_dataloader, val_dataloader, test_dataloader):
    save_path = "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/temp_results/experiment_2/dataset_representation.txt"
    with open(save_path, 'w') as f:
        # Redirect standard output to the file
        sys.stdout = f
        train_dataloader = create_dataloader(train_dataloader)
        val_dataloader = create_dataloader(val_dataloader)
        test_dataloader = create_dataloader(test_dataloader)
        print("train dataset:")
        count_labels_representation_in_dataset(train_dataloader.dataset)
        print("\n")
        print("val dataset:")
        count_labels_representation_in_dataset(val_dataloader.dataset)
        print("\n")
        print("test dataset:")
        count_labels_representation_in_dataset(test_dataloader.dataset)


def count_labels_representation_in_dataset(dataset):
    labels = []
    for i in range(len(dataset)):
        labels += list(dataset[i]['data_samples'].gt_instances.labels.numpy())

    counter = Counter(labels)
    labels_to_classes_names = dataset.METAINFO['classes']
    counter_dict={}
    for label, count in counter.items():
        print(f"{labels_to_classes_names[label]}: {count}")
        counter_dict[labels_to_classes_names[label]] = count
    return counter_dict


def copy_files_according_to_horizontal_dataset():
    source_images_train_path = "/home/shoval/Documents/Repositories/data/gsd_normalized_dataset_rotated/train_all/images"
    source_labels_train_path = "/home/shoval/Documents/Repositories/data/gsd_normalized_dataset_rotated/train_all/labelTxt"
    dest_images_val_path = "/home/shoval/Documents/Repositories/data/gsd_normalized_dataset_rotated/val/images"
    dest_labels_val_path = "/home/shoval/Documents/Repositories/data/gsd_normalized_dataset_rotated/val/labelTxt"
    dest_images_train_path = "/home/shoval/Documents/Repositories/data/gsd_normalized_dataset_rotated/train/images"
    dest_labels_train_path = "/home/shoval/Documents/Repositories/data/gsd_normalized_dataset_rotated/train/labelTxt"

    instruction_path = "/home/shoval/Documents/Repositories/data/gsd_normalized_dataset/val/images"
    i = 0
    for file_name in os.listdir(instruction_path):
        name, extension = os.path.splitext(file_name)
        image_to_copy = os.path.join(source_images_train_path, name + ".png")
        if os.path.isfile(image_to_copy):
            i += 1
            shutil.copy(image_to_copy, dest_images_val_path)
            shutil.copy(os.path.join(source_labels_train_path, name + ".txt"), dest_labels_val_path)
    print(f"{i} files were copied")


def calculate_area_size_statistics_for_each_class(train_dataloader, val_dataloader, test_dataloader):
    pixel_to_meter_squared_factor = 4
    save_path = "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/temp_results/experiment_2/area_size_statistics.json"
    sizes_file_path = "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/temp_results/experiment_2/classes_sizes.txt"
    train_dataloader = create_dataloader(train_dataloader)
    val_dataloader = create_dataloader(val_dataloader)
    test_dataloader = create_dataloader(test_dataloader)
    if not os.path.exists(save_path):
        areas_dict = defaultdict(list)
        for i in range(len(train_dataloader.dataset)):
            for area, label in zip(train_dataloader.dataset[i]['data_samples'].gt_instances.bboxes.areas.numpy(),
                                   train_dataloader.dataset[i]['data_samples'].gt_instances.labels.numpy()):
                areas_dict[int(label)].append(area/pixel_to_meter_squared_factor)

        for i in range(len(val_dataloader.dataset)):
            for area, label in zip(val_dataloader.dataset[i]['data_samples'].gt_instances.bboxes.areas.numpy(),
                                   val_dataloader.dataset[i]['data_samples'].gt_instances.labels.numpy()):
                areas_dict[int(label)].append(area/pixel_to_meter_squared_factor)

        for i in range(len(test_dataloader.dataset)):
            for area, label in zip(test_dataloader.dataset[i]['data_samples'].gt_instances.bboxes.areas.numpy(),
                                   test_dataloader.dataset[i]['data_samples'].gt_instances.labels.numpy()):
                areas_dict[int(label)].append(area/pixel_to_meter_squared_factor)

        with open(save_path, 'w') as file:
            json.dump(areas_dict, file, indent=4)
    else:
        with open(save_path, 'r') as file:
            areas_dict = json.load(file)

    means = []
    stds = []
    medians = []
    with open(sizes_file_path, 'w') as file:
        for label, areas in areas_dict.items():
            median = statistics.median(areas)
            medians.append(median)
            mean = statistics.mean(areas)
            means.append(mean)
            std = np.std(areas)
            stds.append(std)
            print(f"Class {train_dataloader.dataset.METAINFO['classes'][int(label)]}: mean area size: {mean}", file=file)
            print(f"Class {train_dataloader.dataset.METAINFO['classes'][int(label)]}: std dev area size: {std}", file=file)
            print(f"Class {train_dataloader.dataset.METAINFO['classes'][int(label)]}: median area size: {median}", file=file)

    class_names = [train_dataloader.dataset.METAINFO['classes'][int(label)] for label in areas_dict.keys()]
    plt.figure(figsize=(10, 10))
    plt.boxplot(list(areas_dict.values()), patch_artist=True)
    plt.title('Average area size statistics for each class')
    plt.xlabel('Classes')
    plt.ylabel('Area means [$meter^2$]')
    plt.yscale('log')
    plt.xticks(list(range(1, len(areas_dict)+1)), class_names, rotation=90)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(
        "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/temp_results/experiment_2/box_plot_statistics.png")


    plt.clf()
    plt.figure(figsize=(15, 10))
    plt.errorbar(list(range(len(means))), means, yerr=stds, fmt='o')

    plt.xlabel('Classes')
    plt.ylabel('Area means')
    plt.xticks(list(range(len(means))), class_names, rotation=90)
    # plt.yscale('log')
    plt.title('Area size statistics for each class')
    plt.tight_layout()
    # Show the plot
    plt.savefig("/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/temp_results/experiment_2/area_size_statistics.png")

    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.scatter(list(range(len(means))), means)
    plt.xlabel('Classes')
    plt.ylabel('Area means [meters^2]')
    plt.yscale('log')
    plt.tight_layout()
    plt.xticks(list(range(len(means))), class_names, rotation=90)
    plt.title('Average area size statistics for each class')
    plt.savefig("/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/temp_results/experiment_2/average_area_size_statistics.png")

    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.scatter(list(range(len(medians))), medians)
    plt.xlabel('Classes')
    plt.ylabel('Area meidans [meters^2]')
    plt.yscale('log')
    plt.tight_layout()
    plt.xticks(list(range(len(medians))), class_names, rotation=90)
    plt.title('Median area size statistics for each class')
    plt.savefig(
        "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/temp_results/experiment_2/median_area_size_statistics.png")



def calculate_data_statistics(train_dataloader, val_dataloader, test_dataloader):
    calculate_area_size_statistics_for_each_class(train_dataloader, val_dataloader, test_dataloader)
    print("Dataset representation\n")
    calculate_dataset_representation(train_dataloader, val_dataloader, test_dataloader)

def main():
    parser = argparse.ArgumentParser(description='Split train dataset to train and val datasets')
    parser.add_argument("--config", type=str, required=True, help="The input dataset path")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    train_dataset_path = config["save_dir"]
    split_train_to_train_and_val_datasets(train_dataset_path)

if __name__ == '__main__':
    main()
    # cfg = Config.fromfile("./configs/experiment_2/config.py")
    # anomaly_detector_cfg = cfg.get("anomaly_detector_cfg")
    # calculate_data_statistics(train_dataloader=anomaly_detector_cfg.train_dataloader,
    #                           val_dataloader=anomaly_detector_cfg.val_dataloader,
    #                           test_dataloader=anomaly_detector_cfg.test_dataloader)
