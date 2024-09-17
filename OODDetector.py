import json
import os
import pickle
import shutil
from collections import defaultdict

from mmdet.registry import TASK_UTILS
import cv2
import torch
import numpy as np
from enum import Enum

from PIL import Image
from matplotlib import pyplot as plt
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F
from detectors.vim import ViM

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

    def score_samples(self, dataloader, save_outliers=False):
        save_data = defaultdict(dict)
        cache_dir = os.path.join(self.test_output, f"{self.ood_type}_scores")
        os.makedirs(cache_dir, exist_ok=True)
        next_cache_index = 0
        for batch_ind, batch in tqdm(enumerate(dataloader)):
            img_names = [os.path.basename(path).split('.')[0] for path in batch['path']]
            images = batch['pixel_values']
            labels = batch['labels']
            curr_cache_index = batch_ind // 1000
            curr_cache_file = os.path.join(cache_dir, f'all_cache_{curr_cache_index}.pkl')
            if os.path.exists(curr_cache_file):
                next_cache_index = curr_cache_index + 1
                continue

            if curr_cache_index != next_cache_index:
                print(next_cache_index)
                with open(os.path.join(cache_dir, f'all_cache_{next_cache_index}.pkl'), 'wb') as f:
                    pickle.dump(save_data, f)
                save_data = defaultdict(dict)
                next_cache_index = curr_cache_index

            scores, pred_labels = self.calculate_method_scores(images.to(device))
            save_data.update({img_name: {"score": score.item(), "label": label.item(), "pred": pred.item()} for score, label, pred, img_name in zip(scores, labels, pred_labels, img_names)})

        # cache leftovers:
        curr_cache_file = os.path.join(cache_dir, f'all_cache_{curr_cache_index}.pkl')
        if not os.path.exists(curr_cache_file):
            with open(os.path.join(cache_dir, f'all_cache_{curr_cache_index}.pkl'), 'wb') as f:
                pickle.dump(save_data, f)

        all_cache = defaultdict(lambda: {"score": None, "label": None, "pred": None})
        all_caches = [x for x in os.listdir(cache_dir) if x.startswith('all_cache_')]
        all_caches.sort(key=lambda x: int(x.split('all_cache_')[-1].split('.pkl')[0]))
        for cache_name in all_caches:
            with open(os.path.join(cache_dir, cache_name), 'rb') as f:
                all_cache.update(pickle.load(f))

        return all_cache

    def train(self):
        pass


class ODINOODDetector(OODDetector):
    def __init__(self, output_dir, current_run_name, epsilon=DEFAULT_ODIN_NOISE_MAGNITUDE,
                 temperature=DEFAULT_ODIN_TEMP, ):
        super().__init__(output_dir, current_run_name)
        self.model = None
        self.epsilon = epsilon
        self.temperature = temperature
        self.ood_type = "odin"
        self.test_output = os.path.join(self.test_output, self.ood_type)
        self.test_dataset_scores_and_labels = os.path.join(self.test_output,
                                                           f"test_dataset_scores_and_labels_{self.ood_type}.json")
        self.outliers_path = os.path.join(self.test_output, "outliers")
        os.makedirs(self.outliers_path, exist_ok=True)

    def calculate_method_scores(self, inputs):
        criterion = nn.CrossEntropyLoss()
        inputs = Variable(inputs, requires_grad=True)
        inputs = inputs.cuda()
        inputs.retain_grad()

        outputs = self.model(inputs)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

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
        return torch.from_numpy(scores), torch.from_numpy(maxIndexTemp)


class EnergyOODDetector(OODDetector):
    def __init__(self, output_dir, current_run_name, epsilon=DEFAULT_ODIN_NOISE_MAGNITUDE,
                 temperature=DEFAULT_ODIN_TEMP, ):
        super().__init__(output_dir, current_run_name)
        self.model = None
        self.epsilon = epsilon
        self.temperature = temperature
        self.ood_type = "energy"
        self.test_output = os.path.join(self.test_output, self.ood_type)
        self.test_dataset_scores_and_labels = os.path.join(self.test_output,
                                                           f"test_dataset_scores_and_labels_{self.ood_type}.json")
        self.outliers_path = os.path.join(self.test_output, "outliers")
        os.makedirs(self.outliers_path, exist_ok=True)

    def calculate_method_scores(self, inputs):
        preds = self.model(inputs)
        scores = self.temperature * torch.log(
            torch.sum(torch.exp(preds.detach().cpu().type(torch.DoubleTensor)) / self.temperature, dim=1))

        return (scores, torch.from_numpy(np.argmax(preds.data.cpu().numpy(), axis=1)))


class MSPOODDetector(OODDetector):
    def __init__(self, output_dir, current_run_name, epsilon=DEFAULT_ODIN_NOISE_MAGNITUDE,
                 temperature=DEFAULT_ODIN_TEMP, ):
        super().__init__(output_dir, current_run_name)
        self.model = None
        self.epsilon = epsilon
        self.temperature = temperature
        self.ood_type = "msp"
        self.test_output = os.path.join(self.test_output, self.ood_type)
        self.test_dataset_scores_and_labels = os.path.join(self.test_output,
                                                           f"test_dataset_scores_and_labels_{self.ood_type}.json")
        self.outliers_path = os.path.join(self.test_output, "outliers")
        os.makedirs(self.outliers_path, exist_ok=True)

    def calculate_method_scores(self, inputs):
        preds = self.model(inputs)
        scores = torch.max(F.softmax(preds, dim=1).detach().cpu(), axis=1)

        return (scores[0], torch.from_numpy(np.argmax(preds.data.cpu().numpy(), axis=1)))


class ViMOODDetector(OODDetector):
    def __init__(self, output_dir, current_run_name, epsilon=DEFAULT_ODIN_NOISE_MAGNITUDE,
                 temperature=DEFAULT_ODIN_TEMP, ):
        super().__init__(output_dir, current_run_name)
        self.model = None
        self.epsilon = epsilon
        self.temperature = temperature
        self.ood_type = "vim"
        self.test_output = os.path.join(self.test_output, self.ood_type)
        self.test_dataset_scores_and_labels = os.path.join(self.test_output,
                                                           f"test_dataset_scores_and_labels_{self.ood_type}.json")
        self.outliers_path = os.path.join(self.test_output, "outliers")
        os.makedirs(self.outliers_path, exist_ok=True)

    def initiate_vim(self, model, train_dataloader):
        self.model = model
        self.vim = ViM(model, )
        train_samples_batch = next(iter(train_dataloader))  # batch of samples from train
        images = train_samples_batch['pixel_values'].to('cuda')
        self.vim.start()
        self.vim.update(images)
        self.vim.end()

    def calculate_method_scores(self, inputs):
        scores = self.vim(inputs).detach().cpu()
        preds = self.model(inputs)

        return (scores, torch.from_numpy(np.argmax(preds.data.cpu().numpy(), axis=1)))


class DOTAv2Dataset:
    pass


def save_TT_1_images(all_scores, all_labels, dataloader, path, logger, abnormal_labels,
                     hashmap_locations_and_anomaly_scores_test_file, visualizer,
                     test_dataset):
    # map = {4: 6, 6: 5, 3: 9, 5: 11, 7: 12, 2: 13}
    # map = {7:0, 11:2, 10:12, 9:14, 8:5, 6:11, 5:4, 4:9, 3:13, 2:7}
    classes = [dataloader.dataset.labels_to_classes_names[i] for i in range(len(dataloader.dataset.labels_to_classes_names))]
    visualizer.dataset_meta = {'classes': classes,
                               'palette': test_dataset.METAINFO['palette'][:len(dataloader.dataset.class_to_idx.keys())]}
    with open(hashmap_locations_and_anomaly_scores_test_file, 'r') as f:
        hashmap_locations_and_anomaly_scores_test = json.load(f)

    save_path = os.path.join(path, f"TT-1_images_examples")
    os.makedirs(save_path, exist_ok=True)
    sorted_scores, sorted_scores_indices = torch.sort(all_scores)
    ood_scores = all_scores[[label in abnormal_labels for label in all_labels]]
    sorted_ood_scores, sorted_ood_scores_indices = torch.sort(ood_scores)
    ood_ranks_in_sorted_scores = torch.searchsorted(sorted_scores, sorted_ood_scores)
    ood_lowest_scores_indices = torch.index_select(sorted_scores_indices, 0, ood_ranks_in_sorted_scores)
    for sample_ind, sample_rank in tqdm(zip(ood_lowest_scores_indices, ood_ranks_in_sorted_scores)):
        sample = dataloader.dataset.__getitem__(sample_ind)
        label = sample['labels']
        if label in abnormal_labels:
            img_id = os.path.basename(sample['path']).split('.')[0]
            original_image_name = img_id[:img_id.rfind("_")]
            all_ood_polys = []
            all_ood_labels = []
            all_ranks = []

            for original_rank, ood_tagged_ind in enumerate(sorted_scores_indices[:sample_rank+1]):
                curr_sample = dataloader.dataset.__getitem__(ood_tagged_ind)
                img_id = os.path.basename(curr_sample['path']).split('.')[0]
                curr_original_img_id = os.path.basename(curr_sample['path']).split('.')[0][:img_id.rfind("_")]
                if curr_original_img_id == original_image_name:
                    curr_file_data = hashmap_locations_and_anomaly_scores_test[img_id]
                    proposals_poly = curr_file_data['poly']
                    all_ood_polys.append(proposals_poly)
                    all_ood_labels.append(curr_sample['labels'])
                    all_ranks.append(original_rank)

            for idx, ood_ind in enumerate(ood_lowest_scores_indices):
                curr_sample = dataloader.dataset.__getitem__(ood_ind)
                img_id = os.path.basename(curr_sample['path']).split('.')[0]
                curr_original_img_id = os.path.basename(curr_sample['path']).split('.')[0][:img_id.rfind("_")]
                curr_rank = ood_ranks_in_sorted_scores[idx]
                if curr_original_img_id == original_image_name and curr_sample['labels'] != 0 and curr_rank not in all_ranks:
                    curr_file_data = hashmap_locations_and_anomaly_scores_test[img_id]
                    proposals_poly = curr_file_data['poly']
                    all_ood_polys.append(proposals_poly)
                    all_ood_labels.append(curr_sample['labels'])
                    all_ranks.append(curr_rank.item())


            # from the patch we get to original image
            sample_idx = test_dataset.get_index_img_id(original_image_name)
            original_sample = test_dataset.__getitem__(sample_idx)['data_samples']
            original_img = cv2.imread(original_sample.img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            preds = InstanceData()
            preds.bboxes = torch.tensor(all_ood_polys)
            preds.scores = torch.tensor(list(range(1, len(all_ood_polys) + 1)))
            preds.labels = torch.tensor(all_ood_labels)
            original_sample.pred_instances = preds
            visualizer.add_datasample(name='result',
                                      image=original_img,
                                      data_sample=original_sample,
                                      draw_gt=False,
                                      out_file=os.path.join(save_path,
                                                            original_image_name + "_in_original_image.png"),
                                      wait_time=0,
                                      draw_text=True)
            logger.info(f"Original ranks are {all_ranks} for image {original_image_name}\n")


def show_objects_misclassifed_by_the_dataset(all_scores, dataloader, path, logger,
                                             hashmap_locations_and_anomaly_scores_test_file, visualizer, test_dataset):
    map = {4: 6, 6: 5, 3: 9, 5: 11, 7: 12, 2: 13, 0:18}
    visualizer.dataset_meta = test_dataset.METAINFO
    visualizer.dataset_meta['classes'] = visualizer.dataset_meta['classes'] + ('background',)
    visualizer.dataset_meta['palette'] = visualizer.dataset_meta['palette'] + [(75, 0, 130)]
    with open(hashmap_locations_and_anomaly_scores_test_file, 'r') as f:
        hashmap_locations_and_anomaly_scores_test = json.load(f)
    save_path = os.path.join(path, f"objects_misclassified_by_dota")
    os.makedirs(save_path, exist_ok=True)
    sorted_scores, sorted_scores_indices = torch.sort(all_scores)
    for sample_rank, sample_ind in tqdm(enumerate(sorted_scores_indices)):
        if sample_rank in [702,703,704]:
            sample = dataloader.dataset.__getitem__(sample_ind)
            label = map[sample['labels']]
            img_id = os.path.basename(sample['path']).split('.')[0]
            original_image_name = img_id[:img_id.rfind("_")]
            curr_file_data = hashmap_locations_and_anomaly_scores_test[img_id]
            proposals_poly = curr_file_data['poly']
            sample_idx = test_dataset.get_index_img_id(original_image_name)
            original_sample = test_dataset.__getitem__(sample_idx)['data_samples']
            original_img = cv2.imread(original_sample.img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_path, f"{original_image_name}_original_image.png"), original_img)
            preds = InstanceData()
            preds.bboxes = torch.tensor(proposals_poly).unsqueeze(0)
            preds.scores = torch.tensor(list(range(1, len(preds.bboxes) + 1)))
            preds.labels = torch.tensor([label])
            original_sample.pred_instances = preds
            visualizer.add_datasample(name='result',
                                      image=original_img,
                                      data_sample=original_sample,
                                      draw_gt=True,
                                      out_file=os.path.join(save_path,
                                                            original_image_name + "_in_original_image.png"),
                                      wait_time=0,
                                      draw_text=False)
            logger.info(f"Original rank is {sample_rank} for image {original_image_name}\n")


def save_k_outliers(all_cache, dataloader, outliers_path, k=50, logger=None):
    all_scores = torch.tensor([img_prop["score"] for img_name, img_prop in all_cache.items()])
    all_labels = torch.tensor([img_prop["label"] for img_name, img_prop in all_cache.items()])
    labels_to_classes_names = dataloader.dataset.labels_to_classes_names
    # Define the transform to unnormalize the image for resnet18 dataloaders
    save_path = os.path.join(outliers_path, f"{k}_lowest_scores_bg_patches")
    os.makedirs(save_path, exist_ok=True)
    lowest_score_outlier_to_relative_path = {}
    highest_score_outlier_to_relative_path = {}

    # save the k lowest scored background samples
    bg_indices = torch.nonzero(all_labels == 0).squeeze()
    bg_scores_filtered = all_scores[bg_indices]
    lowest_k_bg_indices = torch.argsort(bg_scores_filtered)[:k]
    lowest_k_bg_indices_original = bg_indices[lowest_k_bg_indices]
    for i, sample_ind in tqdm(enumerate(lowest_k_bg_indices_original)):
        # save outlier image
        sample = dataloader.dataset.__getitem__(sample_ind)
        original_image_path = os.path.join(dataloader.dataset.root, sample["path"])
        image = Image.open(original_image_path)
        label = sample['labels']
        new_image_path = os.path.join(save_path, f"{labels_to_classes_names[label]}_bg_outlier_num_{i + 1}.jpg")
        image.save(new_image_path, 'JPEG')

    # save the k lowest score samples
    save_path = os.path.join(outliers_path, f"{k}_lowest_scores_patches")
    os.makedirs(save_path, exist_ok=True)
    lowest_k_indices = torch.argsort(all_scores)[:k]
    for i, sample_ind in tqdm(enumerate(lowest_k_indices)):
        # save outlier image
        sample = dataloader.dataset.__getitem__(sample_ind)
        label = sample['labels']
        original_image_path = os.path.join(dataloader.dataset.root, sample["path"])
        image = Image.open(original_image_path)
        new_image_path = os.path.join(save_path, f"{labels_to_classes_names[label]}_outlier_num_{i + 1}.jpg")
        lowest_score_outlier_to_relative_path[f"{labels_to_classes_names[label]}_outlier_num_{i + 1}"] = sample["path"]
        image.save(new_image_path, 'JPEG')

    # save the k highest score samples
    save_path = os.path.join(outliers_path, f"{k}_highest_scores_patches")
    os.makedirs(save_path, exist_ok=True)
    highest_k_indices = torch.argsort(all_scores, descending=True)[:k]
    for i, sample_ind in tqdm(enumerate(highest_k_indices)):
        # save outlier image
        sample = dataloader.dataset.__getitem__(sample_ind)
        label = sample['labels']
        original_image_path = os.path.join(dataloader.dataset.root, sample["path"])
        image = Image.open(original_image_path)
        new_image_path = os.path.join(save_path, f"{labels_to_classes_names[label]}_outlier_num_{i + 1}.jpg")
        highest_score_outlier_to_relative_path[f"{labels_to_classes_names[label]}_outlier_num_{i + 1}"] = sample["path"]
        image.save(new_image_path, 'JPEG')

    # save the k highest scored background samples
    save_path = os.path.join(outliers_path, f"{k}_highest_bg_scores_patches")
    os.makedirs(save_path, exist_ok=True)
    highest_k_bg_indices = torch.argsort(bg_scores_filtered, descending=True)[:k]
    highest_k_bg_indices_original = bg_indices[highest_k_bg_indices]
    for i, sample_ind in tqdm(enumerate(highest_k_bg_indices_original)):
        # save outlier image
        sample = dataloader.dataset.__getitem__(sample_ind)
        label = sample['labels']
        original_image_path = os.path.join(dataloader.dataset.root, sample["path"])
        image = Image.open(original_image_path)
        new_image_path = os.path.join(save_path, f"{labels_to_classes_names[label]}_bg_outlier_num_{i + 1}.jpg")
        image.save(new_image_path, 'JPEG')

    with open(os.path.join(outliers_path, f"{k}_lowest_scores_outliers_to_relative_path.json"), 'w') as f:
        json.dump(lowest_score_outlier_to_relative_path, f, indent=4)

    with open(os.path.join(outliers_path, f"{k}_highest_scores_outliers_to_relative_path.json"), 'w') as f:
        json.dump(highest_score_outlier_to_relative_path, f, indent=4)


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
    distances = torch.norm(ood_tagged_features - highest_score_ood_sample_features, dim=1)  # euclidean distance
    # distances = F.cosine_similarity(ood_tagged_features, highest_score_ood_sample_features.unsqueeze(0), dim=1) # cosine similarity

    ood_high_thresh_distance_scores = distances[[label in ood_labels for label in ood_tagged_labels]]
    sorted_ood_high_thresh_distance_scores, ood_high_thresh_distance_scores_indices = torch.sort(
        ood_high_thresh_distance_scores)

    sorted_high_thresh_distance_scores, sorted_high_thresh_distance_scores_indices = torch.sort(distances)
    ood_ranks_in_sorted_high_thresh_distance_scores = torch.searchsorted(sorted_high_thresh_distance_scores,
                                                                         sorted_ood_high_thresh_distance_scores)
    logger.info("The OOD ranks are:\n")
    logger.info(f"{ood_ranks_in_sorted_high_thresh_distance_scores}")
    plt.figure()
    plt.plot(list(range(len(ood_ranks_in_sorted_high_thresh_distance_scores))),
             torch.sort(ood_ranks_in_sorted_high_thresh_distance_scores)[0])
    plt.xlabel('OOD distance from first ood sample rank')
    plt.ylabel('distance from first ood sample rank')
    plt.title('OOD distance from first ood sample rank')
    plt.yscale('log')
    # plt.xscale('log')
    plt.grid(True)
    plt.savefig(path + f"/OOD_cs_distance_from_first_ood_sample_rank.png")
