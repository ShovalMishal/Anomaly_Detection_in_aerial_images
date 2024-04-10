import os
import shutil

from DOTA_devkit.DOTA import DOTA
from utils import create_dataloader
import random

# def arrange_data():
#     train_dota_obj = DOTA(basepath='/home/shoval/Documents/Repositories/data/gsd_normalized_dataset/train')
#     dest_image_folder = "/home/shoval/Documents/Repositories/data/gsd_normalized_dataset/val/images/"
#     # dest_labels_folder = "/home/shoval/Documents/Repositories/data/gsd_normalized_dataset/val/labelTxt/"
#     # source_image_folder = '/home/shoval/Documents/Repositories/data/gsd_normalized_dataset/train/images'
#     # source_labels_folder = '/home/shoval/Documents/Repositories/data/gsd_normalized_dataset/train/labelTxt'
#     # images_to_copy = []
#     # for class_obj in list(train_dota_obj.catToImgs.keys()):
#     #     images_list = train_dota_obj.catToImgs[class_obj]
#     #     sample_size = int(0.1 * len(images_list))
#     #     sampled_images = random.sample(images_list, sample_size)
#     #     images_to_copy+=sampled_images
#     # images_to_copy = list(set(images_to_copy))
#     # for img_name in images_to_copy:
#     #     shutil.move(os.path.join(source_image_folder, img_name+'.png'), dest_image_folder)
#     #     shutil.move(os.path.join(source_labels_folder, img_name+'.txt'), dest_labels_folder)

def copy_data_and_sample_from_val_background():
    source_data = "/storage/shoval/Anomaly_Detection_in_aerial_images/results/train/anomaly_detection_result/extracted_bboxes_data"
    dest_data =  "/storage/shoval/Anomaly_Detection_in_aerial_images/results/train/anomaly_detection_result/sampled_extracted_bboxes_data"
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


if __name__ == '__main__':
    copy_data_and_sample_from_val_background()



