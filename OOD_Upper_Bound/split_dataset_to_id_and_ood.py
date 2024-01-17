"""This module creates new datasets folder of id and ood datasets."""
import os
import argparse


def create_ood_id_dataset(src_root: str, target_root: str, ood_classes: list):
    # create src root and target root directories
    os.makedirs(target_root, exist_ok=True)
    os.makedirs(os.path.join(target_root, 'id_dataset'), exist_ok=True)
    os.makedirs(os.path.join(target_root, 'ood_dataset'), exist_ok=True)
    classes_path = os.path.join(src_root,"images")
    for _class in os.listdir(classes_path):
        if _class not in ood_classes:
            if not os.path.exists(os.path.join(target_root, 'id_dataset', _class)):
                os.symlink(os.path.join(classes_path, _class), os.path.join(target_root, 'id_dataset', _class),
                           target_is_directory=True)

    for _class in ood_classes:
        if not os.path.exists(os.path.join(target_root, 'ood_dataset', _class)):
            os.symlink(os.path.join(classes_path, _class), os.path.join(target_root, 'ood_dataset', _class),
                       target_is_directory=True)

