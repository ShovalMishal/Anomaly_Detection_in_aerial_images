import json
import os
import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import Compose
from mmrotate.datasets.dota import  DOTAv2Dataset
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def transform_to_imshow(image):
    image = image*std + mean
    image=image*255
    return image.squeeze(dim=0)

def transform_func(image):
    resized_image = TF.resize(image, (224, 224), interpolation=Image.BILINEAR)
    tensor_image = TF.to_tensor(resized_image)
    # normalized_image = tensor_image / 255.0
    normalized_image = (tensor_image - mean) / std
    return normalized_image


transform = Compose([transform_func])


class image_pyramid_patches_dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.metadata_path = os.path.join(self.root_dir, "metadata")
        self.images_dir = os.path.join(self.root_dir, "images")
        self.class_labels = os.listdir(self.images_dir)
        self.transform = transform
        self.labels_to_cls_num = {class_label: ind for ind, class_label in enumerate(DOTAv2Dataset.METAINFO['classes'])}
        self.labels_to_cls_num["background"] = -1
        self.dataset_len = sum([len(os.listdir(os.path.join(self.images_dir, class_label))) for class_label in self.class_labels])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        for class_label in self.class_labels:
            class_dir = os.path.join(self.images_dir, class_label)
            num_images = len(os.listdir(class_dir))
            if index < num_images:
                break
            index -= num_images
        image_name = os.listdir(class_dir)[index]
        image_path = os.path.join(class_dir, image_name)
        # origin_image_name = image_name[:image_name.rfind("_")]
        # patch_name = image_name[:image_name.find(".png")]
        # metadata_path = os.path.join(self.metadata_path, origin_image_name +".json")
        # with open(metadata_path, 'r') as f:
        #     metadata = json.load(f)[patch_name]
        image = self.transform(Image.open(image_path))
        label = self.labels_to_cls_num[class_label]
        return image, label   #, metadata





