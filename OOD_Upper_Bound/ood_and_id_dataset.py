import os
import torch
from torchvision.datasets import ImageFolder
from collections import Counter



class OODAndIDDataset(ImageFolder):
    def __init__(self, root_dir, dataset_type, transform=None, ood_classes_names=[]):
        super(OODAndIDDataset, self).__init__(root=root_dir, transform=transform)
        ood_indices =sorted([self.classes.index(ood_class_name) for ood_class_name in ood_classes_names if
                       ood_class_name in self.classes])
        self.custom_label_mapping = {i: i for i in range(len(self.classes))}
        for i, curr_ind in enumerate(ood_indices):
            value_to_switch = self.custom_label_mapping[curr_ind]
            for j in range(curr_ind + 1, len(self.classes)):
                self.custom_label_mapping.update({j: value_to_switch})
                value_to_switch += 1
            self.custom_label_mapping.update({curr_ind: len(self.classes) - i - 1})
        self.labels_to_classe_names = {self.custom_label_mapping[i]: self.classes[i] for i in range(len(self.classes))}
        self.classes_names_to_labels = {self.classes[i]: self.custom_label_mapping[i] for i in range(len(self.classes))}
        print(f"final mapping for {dataset_type} dataset is {self.custom_label_mapping}\n")
        self.ood_classes = {name: self.classes_names_to_labels[name] for name in ood_classes_names if name in
                            self.classes_names_to_labels}
        # Print number of samples from each class and calculate class weights
        self.weights = {}
        print(f"************** print dataset {dataset_type} class counts ****************\n")
        class_counts = Counter(self.targets)
        weights_sum = 0
        for class_label, count in class_counts.items():
            class_name = self.labels_to_classe_names[self.custom_label_mapping[class_label]]
            self.weights[class_label] = 1 / count
            weights_sum+=1/count
            print(f"Class {class_name}: {count} samples")
        self.weights = {class_label: self.weights[class_label] / weights_sum for class_label, _ in class_counts.items()}
        print("weights are:")
        print(self.weights)
        print("**********************************************************************\n")

    def __getitem__(self, index):
        original_tuple = super(OODAndIDDataset, self).__getitem__(index)
        label = self.custom_label_mapping[original_tuple[1]]
        return original_tuple[0], label
