import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
from tqdm import tqdm

from Embedder import Resnet_Embedder, Embedder
from results import analyze_roc_curve
from utils import remove_empty_folders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_K = 999


class AnomalyDetector:
    def __init__(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


def set_background_to_0(labels, background_label_index):
    labels = torch.where(labels == 0, background_label_index, torch.where(labels == background_label_index, 0,
                                                                          labels))
    return labels


def calculate_scores_accord_k_value(sorted_diff, k):
    closest_values = sorted_diff[:, :k]
    scores = torch.mean(closest_values.float(), dim=1)
    return scores


class KNNAnomalyDetetctor(AnomalyDetector):
    def __init__(self, embedder_cfg, anomaly_detetctor_cfg, output_dir, dataset_cfg):
        self.embedder = Resnet_Embedder() if embedder_cfg.type == "resnet" else Embedder()
        self.k_list = anomaly_detetctor_cfg.k
        self.use_cache = anomaly_detetctor_cfg.use_cache
        self.sample_ratio = anomaly_detetctor_cfg.sample_ratio
        self.output_dir = output_dir
        self.dictionary_path = os.path.join(self.output_dir, 'features_dict.pt')
        self.dataset_cfg = dataset_cfg
        self.train_scores_table_path = os.path.join(self.output_dir, 'train_scores_table.pt')

    def train(self):
        # create train dataset
        dataset_path = os.path.join(self.dataset_cfg["path"], "subtrain", "images")
        from image_pyramid_patches_dataset import transform
        remove_empty_folders(dataset_path)
        patches_dataset = torchvision.datasets.DatasetFolder(dataset_path, extensions=(".png"), transform=transform,
                                                             loader=Image.open)
        background_label_index = patches_dataset.classes.index("background")
        data_loader = DataLoader(patches_dataset, batch_size=self.dataset_cfg["batch_size"],
                                 shuffle=self.dataset_cfg["shuffle"], num_workers=self.dataset_cfg["num_workers"])

        # create dictionary
        if not self.use_cache:
            self.create_dictionary(data_loader, background_label_index)

        # calculate scores table
        data = torch.load(self.dictionary_path)
        scores, labels = self.calculate_scores_and_labels(data["features"], data_loader, background_label_index)
        # save txt files contains the images for OOD dataset and ID dataset

    def create_dictionary(self, data_loader, background_label_index):
        # save dictionary using the loaded embedder
        labels = []
        # preallocate memory
        sample_per_batch = int(data_loader.batch_size * self.sample_ratio)
        features = torch.zeros((len(data_loader) * sample_per_batch, 2048)).cpu().detach()

        for idx, data_batch in tqdm(enumerate(data_loader)):
            images_patches_features, patches_labels = self.embedder.calculate_batch_features_and_labels(data_batch)
            images_patches_features = images_patches_features.cpu()
            # sample sampled_ratio out of the patches
            random_indices = torch.randperm(images_patches_features.size(0))[
                             :int(self.sample_ratio * images_patches_features.size(0))]
            sampled_features = torch.index_select(images_patches_features, 0, random_indices)
            sampled_labels = torch.index_select(patches_labels, 0, random_indices)
            features[idx * sample_per_batch:sample_per_batch * (idx + 1), :] = sampled_features.cpu().detach()
            labels.append(sampled_labels)
        labels = torch.cat(labels, dim=0)
        # background should be labeled as 0
        labels = set_background_to_0(labels, background_label_index)
        data = {'features': features, 'labels': labels}
        torch.save(data, self.dictionary_path)

    def calculate_scores_and_labels(self, dictionary, dataloader, background_label_index):
        # save a table with len_dataset X max_k if not exist, otherwise load it
        scores = []
        labels = []
        dictionary = dictionary.to(device)
        if not os.path.exists(self.train_scores_table_path):
            for idx, data_batch in tqdm(enumerate(dataloader)):
                images_patches_features, batch_labels = self.embedder.calculate_batch_features_and_labels(data_batch)
                batch_scores = self.calculate_scores_table(images_patches_features, dictionary).cpu()
                scores.append(batch_scores)
                labels.append(batch_labels)
            scores = torch.concat(scores, dim=0)
            labels = torch.concat(labels, dim=0)
            labels = set_background_to_0(labels, background_label_index)
            scores_labels_dict = {"scores": scores, "labels": labels}
            torch.save(scores_labels_dict, self.train_scores_table_path)

        # find the best k and calculate scores according to it's val
        scores_labels_dict = torch.load(self.train_scores_table_path)
        scores_table = scores_labels_dict["scores"]
        labels = scores_labels_dict["labels"]
        k_lowest_fpr, lowest_fpr_cal = 0, float("inf")
        relevant_indices_for_chosen_k = []
        fg_indices = torch.nonzero(labels > 0).squeeze().numpy
        for k_value in self.k_list:
            scores = calculate_scores_accord_k_value(scores_table, k_value)
            threshold, chosen_fpr, chosen_tpr, relevant_indices_tpr_95 = analyze_roc_curve(labels, scores, k_value,
                                                                                           desired_tpr=0.95)
            print(f"k = {k_value}: for tpr {chosen_tpr}, the fpr is {chosen_fpr} and the threshold is {threshold}\n")
            if chosen_fpr < lowest_fpr_cal:
                lowest_fpr_cal = chosen_fpr
                k_lowest_fpr = k_value
                relevant_indices_for_chosen_k = np.unique(np.concatenate((relevant_indices_tpr_95, fg_indices)))
        print(f"The k with the lower fpr for tpr 95 is {k_lowest_fpr}\n")

    def calculate_scores_table(self, test_features, train_features):
        diff = torch.cdist(test_features, train_features)
        # Sort the differences along the second dimension
        sorted_diff, indices = torch.sort(diff, dim=1)
        # Select the k closest values
        scores = sorted_diff[:, :MAX_K]
        return scores
