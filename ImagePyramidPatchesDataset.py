import os
import torch
from torchvision.datasets import ImageFolder
from collections import Counter

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def transform_to_imshow(image):
    image = image * std + mean
    image = image * 255
    return image.squeeze(dim=0)


class ImagePyramidPatchesDataset(ImageFolder):
    def __init__(self, root_dir, inclusion_file_path, dataset_type, output_dir, transform=None, is_valid_file=None,
                 ood_classes_names=[]):
        super(ImagePyramidPatchesDataset, self).__init__(root=root_dir, transform=transform,
                                                         is_valid_file=is_valid_file)
        background_ind = self.classes.index("background")
        ood_indices =sorted([self.classes.index(ood_class_name) for ood_class_name in ood_classes_names if
                       ood_class_name in self.classes])
        self.custom_label_mapping = {i: i for i in range(len(self.classes))}
        for i, curr_ind in enumerate(ood_indices):
            value_to_switch = self.custom_label_mapping[curr_ind]
            for j in range(curr_ind + 1, len(self.classes)):
                self.custom_label_mapping.update({j: value_to_switch})
                value_to_switch += 1
            self.custom_label_mapping.update({curr_ind: len(self.classes) - i - 1})
        self.custom_label_mapping.update({background_ind: 0,
                                          list(self.custom_label_mapping.keys())[list(self.custom_label_mapping.values()).index(0)]: self.custom_label_mapping[background_ind]})
        self.labels_to_classe_names = {self.custom_label_mapping[i]: self.classes[i] for i in range(len(self.classes))}
        self.classes_names_to_labels = {self.classes[i]: self.custom_label_mapping[i] for i in range(len(self.classes))}
        print(f"final mapping for {dataset_type} dataset is {self.custom_label_mapping}\n")
        self.ood_classes = {name: self.classes_names_to_labels[name] for name in ood_classes_names if name in
                            self.classes_names_to_labels}
        self.root_dir = root_dir
        self.metadata_path = os.path.join(os.path.dirname(self.root_dir), "metadata")
        self.inclusion_file_path = inclusion_file_path
        self.scores_and_labels_file_ad = os.path.join(output_dir,
                                                      f'train/anomaly_detection_result/{dataset_type}_dataset_scores_and_labels.json')
        self.scores_and_labels_file_ood = os.path.join(output_dir,
                                                       f'test/OOD/{dataset_type}_dataset_scores_and_labels.json')
        # Print number of samples from each class and calculate class weights
        self.class_weight = {}
        print(f"************** print dataset {dataset_type} class counts ****************\n")
        class_counts = Counter(self.targets)
        for class_label, count in class_counts.items():
            class_name = self.labels_to_classe_names[class_label]
            self.class_weight[class_label] = 1/count
            print(f"Class {class_name}: {count} samples")
        print("**********************************************************************\n")

    def __getitem__(self, index):
        original_tuple = super(ImagePyramidPatchesDataset, self).__getitem__(index)
        image_path = self.imgs[index][0]
        ret_dict = {}
        if isinstance(original_tuple[0], torch.Tensor):
            ret_dict['pixel_values'] = original_tuple[0]
        else:
            ret_dict = original_tuple[0]
        # image_name = os.path.basename(image_path)
        # origin_image_name = image_name[:image_name.rfind("_")]
        # patch_name = image_name[:image_name.find(".png")]
        # metadata_path = os.path.join(self.metadata_path, origin_image_name + ".json")
        # with open(metadata_path, 'r') as f:
        #     metadata = json.load(f)[patch_name]
        # tuple_with_metadata = (original_tuple + (metadata,))
        labels = self.custom_label_mapping[original_tuple[1]]
        ret_dict['labels'] = labels
        ret_dict['path_to_image'] = image_path
        return ret_dict
