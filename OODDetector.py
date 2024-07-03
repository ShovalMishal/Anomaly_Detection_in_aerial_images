import os
import pickle

import torch
import numpy as np
from enum import Enum

from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms


DEFAULT_ODIN_TEMP = 1000
DEFAULT_ODIN_NOISE_MAGNITUDE = 1e-3


class OODDatasetType(Enum):
    IN_DISTRIBUTION = 0
    OUT_OF_DISTRIBUTION = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OODDetector:
    def __init__(self, output_dir, current_run_name):
        self.output_dir = output_dir
        self.train_output = os.path.join(self.output_dir, "train/OOD", current_run_name)
        self.test_output = os.path.join(self.output_dir, "test/OOD", current_run_name)
        os.makedirs(self.train_output, exist_ok=True)
        os.makedirs(self.test_output, exist_ok=True)
        self.test_dataset_scores_and_labels = os.path.join(self.test_output, "test_dataset_scores_and_labels.json")
        self.outliers_path = os.path.join(self.test_output, "outliers")
        os.makedirs(self.outliers_path, exist_ok=True)

    def score_samples(self, dataloader, return_background_score=False):
        pass

    def train(self):
        pass


class ODINOODDetector(OODDetector):
    def __init__(self, output_dir, current_run_name, epsilon=DEFAULT_ODIN_NOISE_MAGNITUDE,
                 temperature=DEFAULT_ODIN_TEMP, ):
        super().__init__(output_dir, current_run_name)
        self.model = None
        self.epsilon = epsilon
        self.temperature = temperature

    def calculate_odin_scores(self, inputs, return_background_score=False):
        criterion = nn.CrossEntropyLoss()
        inputs = Variable(inputs, requires_grad=True)
        inputs = inputs.cuda()
        inputs.retain_grad()

        outputs = self.model(inputs)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        if return_background_score:
            background_score = outputs.data.cpu().numpy()[:, 0]

        # Using temperature scaling
        outputs = outputs / self.temperature

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(input=inputs.data, alpha=-self.epsilon, other=gradient)
        outputs = self.model(Variable(tempInputs))
        outputs = outputs / self.temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        scores = np.max(nnOutputs, axis=1)
        return torch.from_numpy(scores), torch.from_numpy(maxIndexTemp), torch.from_numpy(background_score)

    def score_samples(self, dataloader, return_background_score=False, save_outliers=False):
        all_scores = []
        all_labels = []
        all_preds = []
        all_bg_scores = []
        cache_dir =os.path.join(self.test_output, "odin_scores")
        os.makedirs(cache_dir, exist_ok=True)
        next_cache_index = 0
        for batch_ind, batch in tqdm(enumerate(dataloader)):
            images = batch['pixel_values']
            labels = batch['labels']
            curr_cache_index = batch_ind // 1000
            curr_cache_file = os.path.join(cache_dir, f'all_scores_cache_{curr_cache_index}.pkl')
            if os.path.exists(curr_cache_file):
                next_cache_index = curr_cache_index + 1
                continue

            if curr_cache_index != next_cache_index:
                print(next_cache_index)
                with open(os.path.join(cache_dir, f'all_scores_cache_{next_cache_index}.pkl'), 'wb') as f:
                    pickle.dump(all_scores, f)
                with open(os.path.join(cache_dir, f'all_labels_cache_{next_cache_index}.pkl'), 'wb') as f:
                    pickle.dump(all_labels, f)
                with open(os.path.join(cache_dir, f'all_preds_cache_{next_cache_index}.pkl'), 'wb') as f:
                    pickle.dump(all_preds, f)
                with open(os.path.join(cache_dir, f'all_bg_scores_cache_{next_cache_index}.pkl'), 'wb') as f:
                    pickle.dump(all_bg_scores, f)
                all_scores = []
                all_labels = []
                all_preds = []
                all_bg_scores = []
                next_cache_index = curr_cache_index

            scores, pred_labels, bg_scores = self.calculate_odin_scores(images.to(device),
                                                                        return_background_score=return_background_score)
            all_scores.append(scores)
            all_labels.append(labels)
            all_preds.append(pred_labels)
            all_bg_scores.append(bg_scores)

        # cache leftovers:
        curr_cache_file = os.path.join(cache_dir, f'all_scores_cache_{curr_cache_index}.pkl')
        if not os.path.exists(curr_cache_file):
            with open(os.path.join(cache_dir, f'all_scores_cache_{curr_cache_index}.pkl'), 'wb') as f:
                pickle.dump(all_scores, f)
            with open(os.path.join(cache_dir, f'all_labels_cache_{curr_cache_index}.pkl'), 'wb') as f:
                pickle.dump(all_labels, f)
            with open(os.path.join(cache_dir, f'all_preds_cache_{curr_cache_index}.pkl'), 'wb') as f:
                pickle.dump(all_preds, f)
            with open(os.path.join(cache_dir, f'all_bg_scores_cache_{next_cache_index}.pkl'), 'wb') as f:
                pickle.dump(all_bg_scores, f)
        all_scores_caches = [x for x in os.listdir(cache_dir) if x.startswith('all_scores_cache_')]
        all_scores_caches.sort(key=lambda x: int(x.split('all_scores_cache_')[-1].split('.pkl')[0]))
        all_scores = []
        for cache_name in all_scores_caches:
            with open(os.path.join(cache_dir, cache_name), 'rb') as f:
                all_scores.append(pickle.load(f))
        all_scores = sum(all_scores, [])

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

        all_bg_scores_caches = [x for x in os.listdir(cache_dir) if x.startswith('all_bg_scores_cache_')]
        all_bg_scores_caches.sort(key=lambda x: int(x.split('all_bg_scores_cache_')[-1].split('.pkl')[0]))
        all_bg_scores = []
        for cache_name in all_bg_scores_caches:
            with open(os.path.join(cache_dir, cache_name), 'rb') as f:
                all_bg_scores.append(pickle.load(f))
        all_bg_scores = sum(all_bg_scores, [])

        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return all_scores, all_labels, torch.cat(all_preds, dim=0), torch.cat(all_bg_scores, dim=0)

def save_k_outliers(all_scores, all_labels, dataloader, outliers_path, k=50):
    labels_to_classes_names = dataloader.dataset.labels_to_classes_names
    # Define the transform to unnormalize the image for resnet18 dataloaders
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    save_path = os.path.join(outliers_path, f"{k}_lowest_scores_bg_patches")
    os.makedirs(save_path, exist_ok=True)
    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1 / s for s in std]),
        transforms.Normalize(mean=[-m for m in mean], std=[1, 1, 1]),
        transforms.ToPILImage()
    ])

    # save the k lowest scored background samples
    bg_indices = torch.nonzero(all_labels == 0).squeeze()
    bg_scores_filtered = all_scores[bg_indices]
    lowest_k_bg_indices = torch.argsort(bg_scores_filtered)[:k]
    lowest_k_bg_indices_original = bg_indices[lowest_k_bg_indices]
    for i, sample_ind in tqdm(enumerate(lowest_k_bg_indices_original)):
        # save outlier image
        sample = dataloader.dataset.__getitem__(sample_ind)
        image = sample['pixel_values']
        label = sample['labels']
        unnormalized_image = unnormalize(image)
        image_path = os.path.join(save_path, f"{labels_to_classes_names[label]}_bg_outlier_num_{i+1}")
        unnormalized_image.save(f"{image_path}.jpg")

    # save the k lowest score samples
    save_path = os.path.join(outliers_path, f"{k}_lowest_scores_patches")
    os.makedirs(save_path, exist_ok=True)
    lowest_k_indices = torch.argsort(all_scores)[:k]
    for i, sample_ind in tqdm(enumerate(lowest_k_indices)):
        # save outlier image
        sample = dataloader.dataset.__getitem__(sample_ind)
        image = sample['pixel_values']
        label = sample['labels']
        unnormalized_image = unnormalize(image)
        image_path = os.path.join(save_path, f"{labels_to_classes_names[label]}_outlier_num_{i+1}")
        unnormalized_image.save(f"{image_path}.jpg")

    # save the k highest score samples
    save_path = os.path.join(outliers_path, f"{k}_highest_scores_patches")
    os.makedirs(save_path, exist_ok=True)
    highest_k_indices = torch.argsort(all_scores, descending=True)[:k]
    for i, sample_ind in tqdm(enumerate(highest_k_indices)):
        # save outlier image
        sample = dataloader.dataset.__getitem__(sample_ind)
        image = sample['pixel_values']
        label = sample['labels']
        unnormalized_image = unnormalize(image)
        image_path = os.path.join(save_path, f"{labels_to_classes_names[label]}_outlier_num_{i+1}")
        unnormalized_image.save(f"{image_path}.jpg")

    # save the k highest scored background samples
    save_path = os.path.join(outliers_path, f"{k}_highest_bg_scores_patches")
    os.makedirs(save_path, exist_ok=True)
    highest_k_bg_indices = torch.argsort(bg_scores_filtered, descending=True)[:k]
    highest_k_bg_indices_original = bg_indices[highest_k_bg_indices]
    for i, sample_ind in tqdm(enumerate(highest_k_bg_indices_original)):
        # save outlier image
        sample = dataloader.dataset.__getitem__(sample_ind)
        image = sample['pixel_values']
        label = sample['labels']
        unnormalized_image = unnormalize(image)
        image_path = os.path.join(save_path, f"{labels_to_classes_names[label]}_bg_outlier_num_{i+1}")
        unnormalized_image.save(f"{image_path}.jpg")


def rank_samples_accord_features(scores, ood_labels, eer_threshold, model, dataloader, path, logger):
    logger.info("Ranking samples according features\n")
    cache_path = os.path.join(path, "ood_tagged_samples_features_and_labels.pkl")
    if not os.path.exists(cache_path):
        scores = -scores
        ood_tagged_scores = scores[scores >= eer_threshold]
        ood_tagged_indices = np.where(scores >= eer_threshold)[0]
        ood_tagged_indices_sorted = torch.argsort(ood_tagged_scores)
        # highest_score_sample_index = torch.argsort(ood_tagged_scores)[-1]
        # highest_score_sample_index_in_dataloader = ood_tagged_indices[highest_score_sample_index]
        ood_tagged_indices_sorted_in_dataloader = ood_tagged_indices[ood_tagged_indices_sorted]
        ood_tagged_labels = []
        ood_tagged_features = []
        for i, sample_ind in tqdm(enumerate(ood_tagged_indices_sorted_in_dataloader)):
            sample = dataloader.dataset.__getitem__(sample_ind)
            image = sample['pixel_values']
            label = sample['labels']
            with torch.no_grad():
                features = model.pen_ultimate_layer(x=image.unsqueeze(dim=0).to(device)).cpu()
                ood_tagged_features.append(features)
            ood_tagged_labels.append(label)
        ood_tagged_features = torch.cat(ood_tagged_features, dim=0)
        with open(cache_path, 'wb') as f:
            pickle.dump((ood_tagged_features, ood_tagged_labels), f)
    else:
        with open(cache_path, 'rb') as f:
            ood_tagged_features, ood_tagged_labels = pickle.load(f)

    is_ood_sample = np.array([label in ood_labels for label in ood_tagged_labels])
    is_ood_sample_indices = np.where(is_ood_sample)[0]
    highest_score_ood_sample_index = is_ood_sample_indices[-1]
    highest_score_ood_sample_features = ood_tagged_features[highest_score_ood_sample_index, :]
    distances = torch.norm(ood_tagged_features-highest_score_ood_sample_features, dim=1) # euclidean distance
    # distances = F.cosine_similarity(ood_tagged_features, highest_score_ood_sample_features.unsqueeze(0), dim=1) # cosine similarity


    ood_high_thresh_distance_scores = distances[[label in ood_labels for label in ood_tagged_labels]]
    sorted_ood_high_thresh_distance_scores, ood_high_thresh_distance_scores_indices = torch.sort(ood_high_thresh_distance_scores)

    sorted_high_thresh_distance_scores, sorted_high_thresh_distance_scores_indices = torch.sort(distances)
    ood_ranks_in_sorted_high_thresh_distance_scores = torch.searchsorted(sorted_high_thresh_distance_scores, sorted_ood_high_thresh_distance_scores)
    logger.info("The OOD ranks are:\n")
    logger.info(f"{ood_ranks_in_sorted_high_thresh_distance_scores}")
    plt.figure()
    plt.plot(list(range(len(ood_ranks_in_sorted_high_thresh_distance_scores))), torch.sort(ood_ranks_in_sorted_high_thresh_distance_scores)[0])
    plt.xlabel('OOD distance from first ood sample rank')
    plt.ylabel('distance from first ood sample rank')
    plt.title('OOD distance from first ood sample rank')
    plt.yscale('log')
    # plt.xscale('log')
    plt.grid(True)
    plt.savefig(path + f"/OOD_cs_distance_from_first_ood_sample_rank.png")

