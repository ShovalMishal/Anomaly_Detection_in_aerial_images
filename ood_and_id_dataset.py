import os
import torch
from torchvision.datasets import ImageFolder
from collections import Counter



class OODAndIDDataset(ImageFolder):
    def __init__(self, root_dir, dataset_type, transform=None, ood_classes_names=None):
        super(OODAndIDDataset, self).__init__(root=root_dir, transform=transform)
        self.custom_label_mapping, self.labels_to_classes_names = generate_label_mappers(classes=self.classes,
                                                                                        ood_classes_names=ood_classes_names)
        self.classes_names_to_labels = {self.classes[i]: self.custom_label_mapping[i] for i in range(len(self.classes))}
        print(f"final mapping for {dataset_type} dataset is {self.custom_label_mapping}\n")
        self.ood_classes = {name: self.classes_names_to_labels[name] for name in ood_classes_names if name in
                            self.classes_names_to_labels}
        # Print number of samples from each class and calculate class weights
        print(f"************** print dataset {dataset_type} class counts ****************\n")
        self.weights, self.class_weights = generate_weights(custom_label_mapping=self.custom_label_mapping,
                                                            labels_to_classes_names=self.labels_to_classes_names,
                                                            targets=self.targets)
        print("weights are:")
        print(self.class_weights)
        print("**********************************************************************\n")

    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        label = self.custom_label_mapping[original_tuple[1]]
        full_path = self.imgs[index][0]
        relative_path = os.path.relpath(full_path, self.root)
        # return original_tuple[0], label
        return {"pixel_values": original_tuple[0], "labels": label, "path": relative_path}


def generate_label_mappers(classes=None, ood_classes_names=None):
    if ood_classes_names is None:
        ood_classes_names = []
    ood_indices = sorted([classes.index(ood_class_name) for ood_class_name in ood_classes_names if
                          ood_class_name in classes])
    custom_label_mapping = {i: i for i in range(len(classes))}
    for i, curr_ind in enumerate(ood_indices):
        value_to_switch = custom_label_mapping[curr_ind]
        for j in range(curr_ind + 1, len(classes)):
            custom_label_mapping.update({j: value_to_switch})
            value_to_switch += 1
        custom_label_mapping.update({curr_ind: len(classes) - i - 1})
    labels_to_classes_names = {custom_label_mapping[i]: classes[i] for i in range(len(classes))} # says after mapping what is the class
    return custom_label_mapping, labels_to_classes_names


def generate_weights(custom_label_mapping, labels_to_classes_names, targets):
    class_counts = Counter([t for t in targets])
    class_weights = {}
    weights_sum = 0
    for class_label, count in class_counts.items():
        curr_class_label = custom_label_mapping[class_label]
        class_name = labels_to_classes_names[curr_class_label]
        class_weights[class_label] = 1 / count
        weights_sum += 1 / count
        print(f"Class {class_name}: {count} samples")
    class_weights = {
        class_label: class_weights[class_label] / weights_sum
        for class_label, _ in class_counts.items()}
    weights = [class_weights[t] for t in targets]
    print(f"weights are: {class_weights}")
    return weights, class_weights

