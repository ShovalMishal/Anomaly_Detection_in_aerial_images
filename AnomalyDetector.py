import json
import os

import random
import numpy as np
import torch
from tqdm import tqdm
from Embedder import ResnetEmbedder
from ImagePyramidPatchesDataset import transform_to_imshow
from results import analyze_roc_curve, plot_graphs
from utils import show_img, create_patches_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_K = 100
# torch.set_seed(0)

# Set a random seed for PyTorch
seed = 42
torch.manual_seed(seed)

# Set a random seed for NumPy (if you use NumPy alongside PyTorch)
np.random.seed(seed)

# Set a random seed for Python's built-in random module (if needed)
random.seed(seed)
# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    # Set a random seed for the GPU
    torch.cuda.manual_seed_all(seed)


class AnomalyDetector:
    """Abstract class of anomaly detector. It receives output dir path, dataset configuration and embedder configuration"""

    def __init__(self, output_dir, dataset_cfg, embedder_cfg):
        os.makedirs(os.path.join(self.output_dir, "train/anomaly_detection_result"), exist_ok=True)
        self.output_dir = output_dir
        self.dataset_cfg = dataset_cfg
        self.embedder = {'resnet': ResnetEmbedder}[embedder_cfg.type]
        self.embedder = self.embedder(embedder_cfg.embedder_dim)
        # create train, test and validation datasets
        self.subtest_data_loader = create_patches_dataloader(type="subtest", dataset_cfg=dataset_cfg,
                                                             transform=self.embedder.transform,
                                                             output_dir=output_dir)
        self.subtrain_data_loader = create_patches_dataloader(type="subtrain", dataset_cfg=dataset_cfg,
                                                              transform=self.embedder.transform,
                                                              output_dir=output_dir)
        self.subval_dataloader = create_patches_dataloader(type="subval", dataset_cfg=dataset_cfg,
                                                           transform=self.embedder.transform,
                                                           output_dir=output_dir)
        self.dataloaders = {"train": self.subtrain_data_loader,
                            "test": self.subtest_data_loader,
                            "val": self.subval_dataloader}

    def train(self):
        pass

    def test(self):
        pass


def calculate_scores_accord_k_value(sorted_diff, k):
    closest_values = sorted_diff[:, :k]
    scores = torch.mean(closest_values.float(), dim=1)
    return scores


def calculate_sub_scores_table(test_features, train_features):
    with torch.no_grad():
        diff = torch.cdist(test_features, train_features)
        # Sort the differences along the second dimension
        sorted_diff, indices = torch.sort(diff, dim=1)
        # Select the k closest values
        scores = sorted_diff[:, :MAX_K]
    return scores


def calculate_scores(test_features, train_features, k=3):
    diff = torch.cdist(test_features, train_features)
    # Sort the differences along the second dimension
    sorted_diff, indices = torch.sort(diff, dim=1)
    # Select the k closest values
    closest_values = sorted_diff[:, :k]
    scores = torch.mean(closest_values.float(), dim=1)
    return scores


def get_outliers(scores, labels, dataset, output_dir, k, outliers_num=10, dataset_name="validation"):
    min_k_fg_scores_indices_in_dataset = torch.tensor([])
    # save high score bgs and low score fgs
    bg_scores = scores[torch.nonzero(labels == 0).squeeze(dim=1)]
    top_k_bg_scores, top_indices_bg_scores = torch.topk(bg_scores, k=outliers_num)
    top_k_bg_scores_indices_in_dataset = torch.where(top_k_bg_scores.unsqueeze(1) == scores.unsqueeze(0))[1]
    top_k_bg_scores_indices_in_dataset = torch.from_numpy(
        np.intersect1d(top_k_bg_scores_indices_in_dataset, torch.nonzero(labels == 0)))
    fg_scores = scores[torch.nonzero(labels > 0).squeeze(dim=1)]
    if fg_scores.size(0) > 0:
        min_k_fg_scores, min_indices_fg_scores = torch.topk(fg_scores, k=outliers_num, largest=False)
        min_k_fg_scores_indices_in_dataset = torch.where(min_k_fg_scores.unsqueeze(1) == scores.unsqueeze(0))[1]
        min_k_fg_scores_indices_in_dataset = torch.from_numpy(np.intersect1d(min_k_fg_scores_indices_in_dataset, torch.nonzero(labels > 0)))
    # save all outliers
    path = os.path.join(output_dir, f"train/anomaly_detection_result/{dataset_name}_outliers_k{str(k)}")
    os.makedirs(path, exist_ok=True)
    save_outliers(indices=top_k_bg_scores_indices_in_dataset, dataset=dataset, path=path, is_bg=True)
    save_outliers(indices=min_k_fg_scores_indices_in_dataset, dataset=dataset, path=path, is_bg=False)


def save_outliers(indices, dataset, path, is_bg):
    classes = dataset.classes
    for i in range(indices.size(0)):
        sample = dataset[indices[i]]
        label = sample["labels"]
        patch = sample["pixel_values"]
        title = f"{i + 1} highest score bg, class {classes[label]}" if is_bg else \
            f"{i + 1} lowest score fg, class {classes[label]}"
        saved_path = os.path.join(path, f"outlier_bg_{i + 1}") if is_bg else os.path.join(path, f"outlier_fg_{i + 1}")
        show_img(transform_to_imshow(patch), title=title, path=saved_path)


class KNNAnomalyDetector(AnomalyDetector):
    def __init__(self, embedder_cfg, anomaly_detetctor_cfg, dataset_cfg, output_dir):
        super().__init__(output_dir, dataset_cfg, embedder_cfg)
        self.k_list = anomaly_detetctor_cfg.k
        self.use_cache = anomaly_detetctor_cfg.use_cache
        self.sample_ratio = anomaly_detetctor_cfg.sample_ratio
        self.dictionary_path = os.path.join(self.output_dir, 'train/anomaly_detection_result/features_dict.pt')
        self.train_scores_table_path = os.path.join(self.output_dir,
                                                    'train/anomaly_detection_result/train_scores_table.pt')
        self.test_scores_table_path = os.path.join(self.output_dir,
                                                   'test/anomaly_detection_result/test_scores_table.pt')
        self.val_scores_table_path = os.path.join(self.output_dir, 'train/anomaly_detection_result/val_scores_table.pt')

    def train(self):
        torch.cuda.empty_cache()
        print("Starting training pipeline...")
        # create dictionary
        if not self.use_cache:
            print("Creating new dictionary...")
            self.create_dictionary(self.subtrain_data_loader)

        # calculate scores table according to validation set and find the "best" k.
        data = torch.load(self.dictionary_path)
        self.create_score_table(dictionary=data["features"], dataloader=self.subval_dataloader,
                                path=self.val_scores_table_path)
        chosen_k, threshold = self.find_best_k_and_threshold(path=self.val_scores_table_path)
        # save abnormal and fg samples according to chosen k and threshold
        test_scores, test_labels = self.save_all_inclusion_files(k=chosen_k, threshold=threshold, dictionary=data["features"])
        plot_graphs(scores=test_scores, labels=test_labels, chosen_k=chosen_k, output_dir=self.output_dir,
                    dataset_name="subtest")

    # def test(self):
    #     print("Start testing...")
    #     # Create test dataset
    #     dataset_path = os.path.join(self.dataset_cfg["path"], "subval", "images")
    #     remove_empty_folders(dataset_path)
    #     patches_dataset = ImagePyramidPatchesDataset(dataset_path, transform=resnet_transform)
    #     background_label_index = patches_dataset.classes.index("background")
    #     data_loader = DataLoader(patches_dataset, batch_size=self.dataset_cfg["batch_size"],
    #                              shuffle=self.dataset_cfg["shuffle"], num_workers=self.dataset_cfg["num_workers"])
    #     data = torch.load(self.dictionary_path)
    #     scores, labels = self.calculate_scores_and_labels_without_score_table(data['features'], data_loader,
    #                                                                           background_label_index)
    #     plot_graphs(scores, labels, 3, self.output_dir)

    def create_dictionary(self, data_loader):
        # save dictionary using the loaded embedder
        labels = []
        # preallocate memory
        sample_per_batch = int(data_loader.batch_size * self.sample_ratio)
        features = torch.zeros((len(data_loader) * sample_per_batch, self.embedder.embedder_dim)).cpu().detach()
        for idx, data_batch in tqdm(enumerate(data_loader)):
            images_patches_features = self.embedder.calculate_batch_features(data_batch["pixel_values"]).cpu()
            patches_labels = data_batch["labels"]
            # sample sampled_ratio out of the patches
            random_indices = torch.randperm(images_patches_features.size(0))[
                             :sample_per_batch]
            sampled_features = torch.index_select(images_patches_features, 0, random_indices)
            sampled_labels = torch.index_select(patches_labels, 0, random_indices)
            features[idx * sample_per_batch:sample_per_batch * (idx + 1), :] = sampled_features.cpu().detach()
            labels.append(sampled_labels)
        labels = torch.cat(labels, dim=0)
        # background should be labeled as 0
        data = {'features': features, 'labels': labels}
        torch.save(data, self.dictionary_path)

    def create_score_table(self, dictionary, dataloader, path):
        # save a table with len_dataset X max_k if not exist, otherwise load it
        labels = []
        dictionary = dictionary.to(device)
        if not os.path.exists(path):
            print("Calculate score table...")
            # preallocate memory
            scores = torch.zeros((len(dataloader) * dataloader.batch_size, MAX_K)).cpu().detach()
            for idx, data_batch in tqdm(enumerate(dataloader)):
                images_patches_features = self.embedder.calculate_batch_features(data_batch["pixel_values"])
                batch_labels = data_batch["labels"]
                batch_scores = calculate_sub_scores_table(images_patches_features, dictionary).cpu()
                if idx == len(dataloader) - 1:
                    scores[idx * dataloader.batch_size:dataloader.batch_size * idx + images_patches_features.size(0), :] \
                        = batch_scores
                    last_batch_size = images_patches_features.size(0)
                    print("last_batch_size: " + str(last_batch_size))
                else:
                    scores[idx * dataloader.batch_size:dataloader.batch_size * (idx + 1), :] = batch_scores
                labels.append(batch_labels)
            scores = scores[:-(dataloader.batch_size - last_batch_size), :]
            labels = torch.concat(labels, dim=0)
            scores_labels_dict = {"scores": scores, "labels": labels}
            # return scores_labels_dict
            torch.save(scores_labels_dict, path)

    def find_best_k_and_threshold(self, path):
        print("Find the best k value according to the lowest fpr...")
        # find the best k and calculate scores according to it's val
        scores_labels_dict = torch.load(path)
        scores_table = scores_labels_dict["scores"]
        labels = scores_labels_dict["labels"]
        k_lowest_fpr, lowest_fpr_cal = 0, float("inf")
        best_thresh = None
        best_scores = []
        for k_value in self.k_list:
            scores = calculate_scores_accord_k_value(scores_table, k_value)
            print(f"Calculating AuC for k={k_value}...")
            threshold, chosen_fpr, chosen_tpr = analyze_roc_curve(labels, scores, desired_tpr=0.95)
            print(f"k = {k_value}: for tpr {chosen_tpr}, the fpr is {chosen_fpr} and the threshold is {threshold}\n")
            if chosen_fpr < lowest_fpr_cal:
                best_scores = scores
                lowest_fpr_cal = chosen_fpr
                k_lowest_fpr = k_value
                best_thresh = threshold
        print(f"The k with the lower fpr for tpr 95 is {k_lowest_fpr}\n")
        # save outliers for validation dataset
        get_outliers(scores=best_scores, labels=labels, dataset=self.subval_dataloader.dataset,
                     output_dir=self.output_dir, k=k_lowest_fpr, outliers_num=20, dataset_name="validation")
        return k_lowest_fpr, best_thresh

    def create_dataset_inclusion_file(self, k, threshold, dataloader, dictionary, dataset_type,
                                      save_scores=True):
        print(f"Calculate inclusion file, scores and labels for {dataset_type} dataset...")
        if not os.path.exists(dataloader.dataset.scores_and_labels_file):
            scores = []
            labels = []
            paths = []
            dictionary = dictionary.to(device)
            for idx, data_batch in tqdm(enumerate(dataloader)):
                images_patches_features = self.embedder.calculate_batch_features(data_batch["pixel_values"])
                patches_labels = data_batch["labels"]
                paths += data_batch["path_to_image"]
                batch_scores = calculate_scores(images_patches_features, dictionary, k=k).cpu()
                batch_labels = patches_labels.cpu()
                scores.append(batch_scores)
                labels.append(batch_labels)
            scores = torch.concat(scores, dim=0)
            labels = torch.concat(labels, dim=0)
        else:
            with open(dataloader.dataset.scores_and_labels_file, 'r') as file:
                data = json.load(file)
                scores = data['scores']
                labels = data['labels']
                assert int(data['k']) == k, "loaded scores and labels is incorrect!"

        abnormal_indices_accord_thresh = np.where((scores.numpy() >= threshold).astype(int) == 1)[0]
        if "train" in dataset_type:
            # inject only in distribution foreground patches for train dataset!
            ood_labels = list(dataloader.dataset.ood_classes.values())
            fg_id_indices = torch.nonzero((labels > 0) & (torch.tensor([x not in ood_labels for x in labels]))).squeeze().numpy()
            abnormal_indices_accord_thresh = np.unique(np.concatenate((abnormal_indices_accord_thresh, fg_id_indices)))
        relevant_paths_for_training = [paths[i] for i in abnormal_indices_accord_thresh]

        # save inclusion text file
        with open(dataloader.dataset.inclusion_file_path, 'w') as file:
            for string in relevant_paths_for_training:
                file.write(string + "\n")

        # save score and labels for train dataset
        if save_scores and not os.path.exists(dataloader.dataset.scores_and_labels_file):
            with open(dataloader.dataset.scores_and_labels_file, 'w') as f:
                k_dict = {}
                k_dict["k"] = str(k)
                k_dict["scores"] = scores.tolist()
                k_dict["labels"] = labels.tolist()
                json.dump(k_dict, f, indent=4)
        if "test" in dataset_type:
            return scores, labels

    # def calculate_scores_and_labels_without_score_table(self, dictionary, dataloader, background_label_index,
    #                                                     save_results=True, k=3):
    #     scores = []
    #     labels = []
    #     dictionary = dictionary.to(device)
    #     for idx, data_batch in tqdm(enumerate(dataloader)):
    #         # start_time = time.time()
    #         images_patches_features = self.embedder.calculate_batch_features(data_batch["pixel_values"])
    #         # end_time = time.time()
    #         # print("Extracting features time: ", end_time - start_time)
    #         patches_labels = data_batch["labels"]
    #         # start_time = time.time()
    #         batch_scores = calculate_scores(images_patches_features, dictionary, k=k).cpu()
    #         # end_time = time.time()
    #         # print("Calculate score time: ", end_time - start_time)
    #         batch_labels = patches_labels.cpu()
    #         scores.append(batch_scores)
    #         labels.append(batch_labels)
    #
    #     scores = torch.concat(scores, dim=0)
    #     labels = torch.concat(labels, dim=0)
    #     get_outliers(scores=scores, labels=labels, dataset=dataloader.dataset, output_dir=self.output_dir, k=k,
    #                  outliers_num=20)
    #
    #     # save score and labels per k
    #     if save_results:
    #         with open(os.path.join(self.output_dir,
    #                                f'statistics/test/anomaly_detection_result/k_{str(k)}_scores_and_labels.json'),
    #                   'w') as f:
    #             k_dict = {}
    #             k_dict["scores"] = scores.tolist()
    #             k_dict["labels"] = labels.tolist()
    #             json.dump(k_dict, f, indent=4)
    #     return scores, labels

    def save_all_inclusion_files(self, k, threshold, dictionary):
        test_scores = []
        test_labels = []
        for data_type, dataloader in self.dataloaders.items():
            results = self.create_dataset_inclusion_file(k=k, threshold=threshold,
                                                         dataloader=dataloader,
                                                         dictionary=dictionary, dataset_type=data_type)
            if "test" in data_type:
                test_scores, test_labels = results
        return test_scores, test_labels
