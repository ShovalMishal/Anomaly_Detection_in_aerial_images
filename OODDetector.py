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
from webencodings import labels

from detectors.vim import ViM
from Classifier import create_dataloaders, DatasetType
from plot_results import plot_graphs
from utils import threshold_and_retrieve_samples

DEFAULT_ODIN_TEMP = 1000
DEFAULT_ODIN_NOISE_MAGNITUDE = 1e-3


class OODDatasetType(Enum):
    IN_DISTRIBUTION = 0
    OUT_OF_DISTRIBUTION = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OODDetector:
    def __init__(self, cfg, output_dir, dataset_dir, hashmap_locations_and_anomaly_scores_test_file,
                 original_data_path, current_run_name, logger):
        self.ood_labels = None
        self.outliers_path = None
        self.test_dataset_scores_and_labels_filtered = None
        self.test_dataset_scores_and_labels = None
        self.test_dataloader = None
        self.output_dir = output_dir
        self.cfg = cfg
        self.logger=logger
        self.model = None
        self.train_output = os.path.join(self.output_dir, "train/OOD", current_run_name)
        self.test_output = os.path.join(self.output_dir, "test/OOD", current_run_name)
        self.train_dataset_dir = os.path.join(dataset_dir, "train")
        self.val_dataset_dir = os.path.join(dataset_dir, "val")
        self.test_dataset_dir = os.path.join(dataset_dir, "test")
        self.hashmap_locations_and_anomaly_scores_test_file=hashmap_locations_and_anomaly_scores_test_file
        self.original_data_path=original_data_path
        os.makedirs(self.train_output, exist_ok=True)
        os.makedirs(self.test_output, exist_ok=True)

    def score_samples(self, dataloader):
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

    def initialize_detector(self, model, train_transforms, val_transforms, output_dir_train_dataset,
                            output_dir_val_dataset, output_dir_test_dataset, batch_size):
        self.model=model

        train_dataloader, _, self.test_dataloader = create_dataloaders(train_transforms=train_transforms,
                                                   val_transforms=val_transforms,
            data_paths={"train": output_dir_train_dataset,
                        "val": output_dir_val_dataset,
                        "test": output_dir_test_dataset}, dataset_type=DatasetType.NONE,
                        ood_classes_names=self.cfg.ood_class_names,
                        val_batch_size=batch_size)
        self.ood_labels = self.test_dataloader.dataset.ood_classes.values()

        if self.cfg.type == 'vim':
            self.initiate_vim(train_dataloader=train_dataloader)

    def test(self, model, train_transforms, val_transforms, visualizer, dota_test_dataset):
        self.initialize_detector(model, train_transforms, val_transforms, self.train_dataset_dir,
                            self.val_dataset_dir, self.test_dataset_dir, self.cfg.batch_size)

        if not os.path.exists(self.test_dataset_scores_and_labels):
            all_cache = self.score_samples(dataloader=self.test_dataloader)
            with open(self.test_dataset_scores_and_labels, 'w') as f:
                json.dump(all_cache, f, indent=4)
        else:
            with open(self.test_dataset_scores_and_labels, 'r') as file:
                all_cache = json.load(file)

        scores, labels, anomaly_scores, anomaly_scores_conv, original_ind = (threshold_and_retrieve_samples
                                                                               (self.test_dataloader,
                                                                                self.hashmap_locations_and_anomaly_scores_test_file,
                                                                                all_cache, self.original_data_path,
                                                                                self.test_dataset_scores_and_labels_filtered,
                                                                                self.cfg.patches_per_image))

        eer_threshold = plot_graphs(scores=scores, anomaly_scores=anomaly_scores,
                                         anomaly_scores_conv=anomaly_scores_conv, labels=labels,
                                         path=self.test_output, title="OOD stage",
                                         abnormal_labels=list(self.test_dataloader.dataset.ood_classes.values()),
                                         dataset_name="test", ood_mode=True,
                                         labels_to_classes_names=self.test_dataloader.dataset.labels_to_classes_names,
                                         plot_EER=True, logger=self.logger, OOD_method=self.cfg.type)

        if self.cfg.save_outliers:
            self.logger.info(f"Saving {self.cfg.num_of_outliers} Outliers\n")
            save_k_outliers(all_scores=scores, all_labels=labels, all_original_indices=original_ind,
                            dataloader=self.test_dataloader, outliers_path=self.outliers_path,
                            k=self.cfg.num_of_outliers)

        if self.cfg.num_of_TT_1_original_images > 0:
            save_TT_1_images(all_scores=scores, all_labels=labels,  all_original_inds=original_ind,
                             dataloader=self.test_dataloader, path=self.test_output, logger=self.logger,
                             abnormal_labels=list(self.ood_labels),
                             hashmap_locations_and_anomaly_scores_test_file=self.hashmap_locations_and_anomaly_scores_test_file,
                             visualizer=visualizer,
                             test_dataset=dota_test_dataset,
                             num_of_TT_1_original_images=self.cfg.num_of_TT_1_original_images)

        if self.cfg.rank_accord_features:
            self.rank_samples_accord_features(scores, original_ind, eer_threshold, self.test_dataloader)


    def rank_samples_accord_features(self, scores, original_ind, eer_threshold, dataloader):
        self.logger.info("Ranking samples according features\n")
        cache_path = os.path.join(self.test_output, "ood_tagged_samples_features_and_labels.pkl")
        if not os.path.exists(cache_path):
            scores = -scores
            ood_tagged_scores = scores[scores >= eer_threshold]
            ood_tagged_original_ind = original_ind[scores >= eer_threshold]
            ood_tagged_indices_sorted = torch.argsort(ood_tagged_scores)
            ood_tagged_indices_sorted_in_dataloader = ood_tagged_original_ind[ood_tagged_indices_sorted]
            ood_tagged_labels = []
            ood_tagged_features = []
            for i, sample_ind in tqdm(enumerate(ood_tagged_indices_sorted_in_dataloader)):
                sample = dataloader.dataset.__getitem__(sample_ind)
                image = sample['pixel_values']
                label = sample['labels']
                with torch.no_grad():
                    features = self.model.pen_ultimate_layer(x=image.unsqueeze(dim=0).to(device)).cpu()
                    ood_tagged_features.append(features)
                ood_tagged_labels.append(label)
            ood_tagged_features = torch.cat(ood_tagged_features, dim=0)
            with open(cache_path, 'wb') as f:
                pickle.dump((ood_tagged_features, ood_tagged_labels), f)
        else:
            with open(cache_path, 'rb') as f:
                ood_tagged_features, ood_tagged_labels = pickle.load(f)

        is_ood_sample = np.array([label in self.ood_labels for label in ood_tagged_labels])
        is_ood_sample_indices = np.where(is_ood_sample)[0]
        highest_score_ood_sample_index = is_ood_sample_indices[-1]
        highest_score_ood_sample_features = ood_tagged_features[highest_score_ood_sample_index, :]
        distances = torch.norm(ood_tagged_features - highest_score_ood_sample_features, dim=1)  # euclidean distance

        ood_high_thresh_distance_scores = distances[[label in self.ood_labels for label in ood_tagged_labels]]
        sorted_ood_high_thresh_distance_scores, ood_high_thresh_distance_scores_indices = torch.sort(
            ood_high_thresh_distance_scores)

        sorted_high_thresh_distance_scores, sorted_high_thresh_distance_scores_indices = torch.sort(distances)
        ood_ranks_in_sorted_high_thresh_distance_scores = torch.searchsorted(sorted_high_thresh_distance_scores,
                                                                             sorted_ood_high_thresh_distance_scores)
        self.logger.info("The OOD ranks are:\n")
        self.logger.info(f"{ood_ranks_in_sorted_high_thresh_distance_scores}")
        plt.figure()
        plt.plot(list(range(len(ood_ranks_in_sorted_high_thresh_distance_scores))),
                 torch.sort(ood_ranks_in_sorted_high_thresh_distance_scores)[0])
        plt.xlabel('OOD distance from first ood sample rank')
        plt.ylabel('distance from first ood sample rank')
        plt.title('OOD distance from first ood sample rank')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(self.test_output + f"/OOD_cs_distance_from_first_ood_sample_rank.pdf")

    def initiate_vim(self, train_dataloader):
        pass
    
class ODINOODDetector(OODDetector):
    def __init__(self, cfg, output_dir, dataset_dir, hashmap_locations_and_anomaly_scores_test_file,
                original_data_path, current_run_name, logger, epsilon=DEFAULT_ODIN_NOISE_MAGNITUDE,
                 temperature=DEFAULT_ODIN_TEMP, ):
        super().__init__(cfg, output_dir, dataset_dir, hashmap_locations_and_anomaly_scores_test_file,
        original_data_path, current_run_name, logger)
        self.model = None
        self.epsilon = epsilon
        self.temperature = temperature
        self.ood_type = "odin"
        self.test_output = os.path.join(self.test_output, self.ood_type)
        self.test_dataset_scores_and_labels = os.path.join(self.test_output,
                                                           f"test_dataset_scores_and_labels_{self.ood_type}.json")
        self.test_dataset_scores_and_labels_filtered = os.path.join(self.test_output,
                                                           f"test_dataset_scores_and_labels_{self.ood_type}_filtered.json")
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
    def __init__(self, cfg, output_dir, dataset_dir, hashmap_locations_and_anomaly_scores_test_file,
                original_data_path, current_run_name, logger, epsilon=DEFAULT_ODIN_NOISE_MAGNITUDE,
                 temperature=DEFAULT_ODIN_TEMP, ):
        super().__init__(cfg, output_dir, dataset_dir, hashmap_locations_and_anomaly_scores_test_file,
        original_data_path, current_run_name, logger)
        self.model = None
        self.epsilon = epsilon
        self.temperature = temperature
        self.ood_type = "energy"
        self.test_output = os.path.join(self.test_output, self.ood_type)
        self.test_dataset_scores_and_labels = os.path.join(self.test_output,
                                                           f"test_dataset_scores_and_labels_{self.ood_type}.json")
        self.test_dataset_scores_and_labels_filtered = os.path.join(self.test_output,
                                                           f"test_dataset_scores_and_labels_{self.ood_type}_filtered.json")
        self.outliers_path = os.path.join(self.test_output, "outliers")
        os.makedirs(self.outliers_path, exist_ok=True)

    def calculate_method_scores(self, inputs):
        preds = self.model(inputs)
        scores = self.temperature * torch.log(
            torch.sum(torch.exp(preds.detach().cpu().type(torch.DoubleTensor)) / self.temperature, dim=1))

        return (scores, torch.from_numpy(np.argmax(preds.data.cpu().numpy(), axis=1)))


class MSPOODDetector(OODDetector):
    def __init__(self, cfg, output_dir, dataset_dir, hashmap_locations_and_anomaly_scores_test_file,
                original_data_path, current_run_name, logger, epsilon=DEFAULT_ODIN_NOISE_MAGNITUDE,
                 temperature=DEFAULT_ODIN_TEMP, ):
        super().__init__(cfg, output_dir, dataset_dir, hashmap_locations_and_anomaly_scores_test_file,
        original_data_path, current_run_name, logger)
        self.model = None
        self.epsilon = epsilon
        self.temperature = temperature
        self.ood_type = "msp"
        self.test_output = os.path.join(self.test_output, self.ood_type)
        self.test_dataset_scores_and_labels = os.path.join(self.test_output,
                                                           f"test_dataset_scores_and_labels_{self.ood_type}.json")
        self.test_dataset_scores_and_labels_filtered = os.path.join(self.test_output,
                                                           f"test_dataset_scores_and_labels_{self.ood_type}_filtered.json")
        self.outliers_path = os.path.join(self.test_output, "outliers")
        os.makedirs(self.outliers_path, exist_ok=True)

    def calculate_method_scores(self, inputs):
        preds = self.model(inputs)
        scores = torch.max(F.softmax(preds, dim=1).detach().cpu(), axis=1)

        return (scores[0], torch.from_numpy(np.argmax(preds.data.cpu().numpy(), axis=1)))


class ViMOODDetector(OODDetector):
    def __init__(self, cfg, output_dir, dataset_dir, hashmap_locations_and_anomaly_scores_test_file,
            original_data_path, current_run_name, logger, epsilon=DEFAULT_ODIN_NOISE_MAGNITUDE,
            temperature=DEFAULT_ODIN_TEMP, ):
        super().__init__(cfg, output_dir, dataset_dir, hashmap_locations_and_anomaly_scores_test_file,
        original_data_path, current_run_name, logger)
        self.model = None
        self.epsilon = epsilon
        self.temperature = temperature
        self.ood_type = "vim"
        self.test_output = os.path.join(self.test_output, self.ood_type)
        self.test_dataset_scores_and_labels = os.path.join(self.test_output,
                                                           f"test_dataset_scores_and_labels_{self.ood_type}.json")
        self.test_dataset_scores_and_labels_filtered = os.path.join(self.test_output,
                                                           f"test_dataset_scores_and_labels_{self.ood_type}_filtered.json")
        self.outliers_path = os.path.join(self.test_output, "outliers")
        os.makedirs(self.outliers_path, exist_ok=True)

    def initiate_vim(self, train_dataloader):
        self.vim = ViM(self.model, )
        train_samples_batch = next(iter(train_dataloader))  # batch of samples from train
        images = train_samples_batch['pixel_values'].to('cuda')
        self.vim.start()
        self.vim.update(images)
        self.vim.end()

    def calculate_method_scores(self, inputs):
        scores = self.vim(inputs).detach().cpu()
        preds = self.model(inputs)

        return (scores, torch.from_numpy(np.argmax(preds.data.cpu().numpy(), axis=1)))


def save_TT_1_images(all_scores, all_labels, all_original_inds, dataloader, path, logger, abnormal_labels,
                     hashmap_locations_and_anomaly_scores_test_file, visualizer,
                     test_dataset, num_of_TT_1_original_images=100):
    classes = [dataloader.dataset.labels_to_classes_names[i] for i in range(len(dataloader.dataset.labels_to_classes_names))]
    visualizer.dataset_meta = {'classes': classes,
                               'palette': test_dataset.METAINFO['palette'][:len(dataloader.dataset.class_to_idx.keys())]}
    with open(hashmap_locations_and_anomaly_scores_test_file, 'r') as f:
        hashmap_locations_and_anomaly_scores_test = json.load(f)

    save_path = os.path.join(path, f"TT-1_images_examples")
    os.makedirs(save_path, exist_ok=True)
    ood_scores = all_scores[[label in abnormal_labels for label in all_labels]]
    ood_original_inds = all_original_inds[[label in abnormal_labels for label in all_labels]]
    sorted_ood_scores, sorted_ood_scores_indices = torch.sort(ood_scores)
    ood_lowest_scores_indices = ood_original_inds[sorted_ood_scores_indices]
    sorted_scores, sorted_scores_ind = torch.sort(all_scores)
    ood_ranks_in_sorted_scores = torch.where(torch.isin(all_labels[sorted_scores_ind], torch.tensor(abnormal_labels)))[0]
    saved_images_num=0
    for sample_ind, sample_rank in tqdm(zip(ood_lowest_scores_indices, ood_ranks_in_sorted_scores)):
        if saved_images_num == num_of_TT_1_original_images:
            break
        sample = dataloader.dataset.__getitem__(sample_ind)
        label = sample['labels']

        if label in abnormal_labels:
            img_id = os.path.basename(sample['path']).split('.')[0]
            original_image_name = img_id[:img_id.rfind("_")]
            all_ood_polys = []
            all_ood_labels = []
            all_ranks = []

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
                                      draw_gt=True,
                                      out_file=os.path.join(save_path,
                                                            original_image_name + "_in_original_image.png"),
                                      wait_time=0,
                                      draw_text=True)
            logger.info(f"Original ranks are {all_ranks} for image {original_image_name}\n")
            saved_images_num+=1


def show_objects_misclassifed_by_the_dataset(all_scores, original_ind, dataloader, path, logger,
                                             hashmap_locations_and_anomaly_scores_test_file, visualizer, test_dataset):
    visualizer.dataset_meta = test_dataset.METAINFO
    visualizer.dataset_meta['classes'] = visualizer.dataset_meta['classes'] + ('background',)
    visualizer.dataset_meta['palette'] = visualizer.dataset_meta['palette'] + [(75, 0, 130)]
    with open(hashmap_locations_and_anomaly_scores_test_file, 'r') as f:
        hashmap_locations_and_anomaly_scores_test = json.load(f)
    save_path = os.path.join(path, f"objects_misclassified_by_dota")
    os.makedirs(save_path, exist_ok=True)
    sorted_scores, sorted_scores_indices = torch.sort(all_scores)
    sorted_original_ind=original_ind[sorted_scores_indices]
    mapping_labels = {label:visualizer.dataset_meta['classes'].index(class_name) for label, class_name in dataloader.dataset.labels_to_classes_names.items()}

    for sample_rank, sample_ind in tqdm(enumerate(sorted_original_ind)):
        if sample_rank in [169, 155, 121, 161]:
            sample = dataloader.dataset.__getitem__(sample_ind)
            label = mapping_labels[sample['labels']]
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


def save_outlier_images(indices, dataloader, save_path, labels_to_classes_names, outlier_type, outlier_dict=None):
    os.makedirs(save_path, exist_ok=True)
    for i, sample_ind in tqdm(enumerate(indices), desc=f"Saving {outlier_type}"):
        sample = dataloader.dataset.__getitem__(sample_ind)
        label = sample['labels']
        original_image_path = os.path.join(dataloader.dataset.root, sample["path"])
        image = Image.open(original_image_path)
        new_image_path = os.path.join(save_path, f"{labels_to_classes_names[label]}_{outlier_type}_num_{i + 1}.jpg")
        image.save(new_image_path, 'JPEG')
        if outlier_dict is not None:
            outlier_dict[f"{labels_to_classes_names[label]}_{outlier_type}_num_{i + 1}"] = sample["path"]


def save_k_outliers(all_scores, all_labels, all_original_indices, dataloader, outliers_path, k=50):
    labels_to_classes_names = dataloader.dataset.labels_to_classes_names
    bg_indices = torch.nonzero(all_labels == 0).squeeze()
    bg_scores_filtered = all_scores[bg_indices]

    lowest_score_outlier_to_relative_path = {}
    highest_score_outlier_to_relative_path = {}

    save_outlier_images(
        all_original_indices[bg_indices[torch.argsort(bg_scores_filtered)[:k]]],
        dataloader,
        os.path.join(outliers_path, f"{k}_lowest_scores_bg_patches"),
        labels_to_classes_names,
        "bg_outlier"
    )

    save_outlier_images(
        all_original_indices[torch.argsort(all_scores)[:k]],
        dataloader,
        os.path.join(outliers_path, f"{k}_lowest_scores_patches"),
        labels_to_classes_names,
        "outlier",
        lowest_score_outlier_to_relative_path
    )

    save_outlier_images(
        all_original_indices[torch.argsort(all_scores, descending=True)[:k]],
        dataloader,
        os.path.join(outliers_path, f"{k}_highest_scores_patches"),
        labels_to_classes_names,
        "outlier",
        highest_score_outlier_to_relative_path
    )

    save_outlier_images(
        all_original_indices[bg_indices[torch.argsort(bg_scores_filtered, descending=True)[:k]]],
        dataloader,
        os.path.join(outliers_path, f"{k}_highest_bg_scores_patches"),
        labels_to_classes_names,
        "bg_outlier"
    )

    with open(os.path.join(outliers_path, f"{k}_lowest_scores_outliers_to_relative_path.json"), 'w') as f:
        json.dump(lowest_score_outlier_to_relative_path, f, indent=4)

    with open(os.path.join(outliers_path, f"{k}_highest_scores_outliers_to_relative_path.json"), 'w') as f:
        json.dump(highest_score_outlier_to_relative_path, f, indent=4)


