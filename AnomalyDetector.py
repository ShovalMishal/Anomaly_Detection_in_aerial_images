import json
import os
from math import ceil
import itertools
import cv2

import random
from typing import Union

from mmdet.structures import DetDataSample

from bbox_regressor import BBoxRegressor
from DOTA_devkit.dota_utils import parse_dota_poly
from PIL import Image
from matplotlib import pyplot as plt
from mmrotate.structures import RotatedBoxes
from mmrotate.structures.bbox import hbox2rbox, qbox2rbox
from mmrotate.evaluation import eval_rbbox_mrecall_for_regressor
import numpy as np
import torch
from mmengine.runner import Runner
from tqdm import tqdm
from single_image_bg_detector.bg_subtraction_with_dino_vit import BGSubtractionWithDinoVit
from DOTA_devkit.DOTA import DOTA
from single_image_bg_detector.bg_subtractor_utils import (extract_patches_accord_heatmap, \
                                                          assign_predicted_boxes_to_gt_boxes_and_save, save_id_gts)
from utils import create_dataloader
from mmengine.structures import InstanceData
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
from mmdet.registry import MODELS, TASK_UTILS


def create_padding_mask(image_path, padding_value):
    orig_img = cv2.imread(image_path)
    mask = np.all(orig_img == padding_value, axis=-1)
    return mask

class AnomalyDetector:
    """Abstract class of anomaly detector. It receives output dir path, dataset configuration and embedder configuration"""

    def __init__(self, output_dir, anomaly_detector_cfg, logger, current_run_name):
        self.anomaly_detector_cfg = anomaly_detector_cfg
        self.logger = logger
        self.test_output_dir = os.path.join(output_dir, "test/anomaly_detection_result", current_run_name)
        self.output_dir = os.path.join(output_dir, "train/anomaly_detection_result", current_run_name)
        self.data_output_dir = os.path.join(self.output_dir,
                                            anomaly_detector_cfg.data_output_dir_name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_output_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
        # create train, test and validation datasets
        self.proposals_cache_path = os.path.join(self.output_dir, 'cache_folder')
        os.makedirs(self.proposals_cache_path, exist_ok=True)

        self.output_dir_train_dataset = os.path.join(self.data_output_dir, "train")
        self.output_dir_val_dataset = os.path.join(self.data_output_dir, "val")
        self.output_dir_test_dataset = os.path.join(self.data_output_dir, "test")
        self.hashmap_locations_and_anomaly_scores_train_file = os.path.join(self.output_dir_train_dataset,
                                                            "hash_map_locations_and_anomaly_scores_train_dataset.json")
        self.hashmap_locations_and_anomaly_scores_val_file = os.path.join(self.output_dir_val_dataset,
                                                            "hash_map_locations_and_anomaly_scores_val_dataset.json")
        self.hashmap_locations_and_anomaly_scores_test_file = os.path.join(self.output_dir_test_dataset,
                                                            "hash_map_locations_and_anomaly_scores_test_dataset.json")
        self.to_extract_patches = anomaly_detector_cfg.extract_patches
        self.evaluate_stage = anomaly_detector_cfg.evaluate_stage
        self.data_path = os.path.dirname(os.path.dirname(anomaly_detector_cfg.train_dataloader.dataset.data_root))

    def run(self):
        pass


class VitBasedAnomalyDetector(AnomalyDetector):
    # extract patches and save them according to their assigned labels
    def __init__(self, anomaly_detector_cfg, output_dir, logger, current_run_name, original_data_path=None,
                 lowest_gsd=None):
        super().__init__(output_dir=output_dir, anomaly_detector_cfg=anomaly_detector_cfg, logger=logger,
                         current_run_name=current_run_name)
        self.logger.info(f"Creating anomaly detector\n")
        self.dota_obj_train = DOTA(basepath=anomaly_detector_cfg.train_dataloader.dataset.data_root)
        self.dota_obj_val = DOTA(basepath=anomaly_detector_cfg.val_dataloader.dataset.data_root)
        self.dota_obj_test = DOTA(basepath=anomaly_detector_cfg.test_dataloader.dataset.data_root)
        self.dino_vit_bg_subtractor = BGSubtractionWithDinoVit(target_dir=self.output_dir,
                                                               vit_patch_size=anomaly_detector_cfg.vit_patch_size,
                                                               vit_arch=anomaly_detector_cfg.vit_arch,
                                                               vit_image_size=anomaly_detector_cfg.vit_image_size,
                                                               dota_obj=self.dota_obj_train,
                                                               threshold=anomaly_detector_cfg.vit_threshold,
                                                               pretrained_weights=anomaly_detector_cfg.pretrained_weights,
                                                               checkpoint_key=anomaly_detector_cfg.checkpoint_key,
                                                               model_mode=anomaly_detector_cfg.vit_model_mode,
                                                               model_type=anomaly_detector_cfg.vit_model_type,
                                                               attention_num=anomaly_detector_cfg.attention_num)
        self.bbox_assigner = TASK_UTILS.build(anomaly_detector_cfg.patches_assigner)
        self.proposals_sizes = anomaly_detector_cfg.proposals_sizes
        self.patches_filtering_threshold = anomaly_detector_cfg.patches_filtering_threshold
        self.original_data_path = original_data_path
        self.skip_stage = anomaly_detector_cfg.skip_stage
        self.bbox_regressor_output_dir = os.path.join(self.output_dir, 'bbox_regressor')
        anomaly_detector_cfg.bbox_regressor['work_dir'] = self.bbox_regressor_output_dir
        self.lowest_gsd = lowest_gsd

    def create_bbox_regressor_runner(self):
        self.anomaly_detector_cfg.bbox_regressor.train_dataloader.dataset.extracted_patches_folder = self.proposals_cache_path
        self.anomaly_detector_cfg.bbox_regressor.val_dataloader.dataset.extracted_patches_folder = self.proposals_cache_path
        self.anomaly_detector_cfg.bbox_regressor.test_dataloader.dataset.extracted_patches_folder = self.proposals_cache_path
        self.bbox_regressor_runner = Runner.from_cfg(self.anomaly_detector_cfg.bbox_regressor)
        self.bbox_regressor_runner.model.initialize(logger=self.logger,
                                                    vit_patch_size=self.anomaly_detector_cfg.vit_patch_size,
                                                    features_extractor=self.dino_vit_bg_subtractor.model,
                                                    dino_vit_bg_subtractor=self.dino_vit_bg_subtractor,
                                                    proposal_sizes=self.proposals_sizes,
                                                    patches_filtering_threshold=self.patches_filtering_threshold,
                                                    output_dir=self.output_dir)

    def extract_patches(self):
        self.logger.info(f"Extract patches\n")
        train_dataloader = create_dataloader(dataloader_cfg=self.anomaly_detector_cfg.train_dataloader)
        val_dataloader = create_dataloader(dataloader_cfg=self.anomaly_detector_cfg.val_dataloader)
        test_dataloader = create_dataloader(dataloader_cfg=self.anomaly_detector_cfg.test_dataloader)
        for dataloader, target_dir in zip([train_dataloader, val_dataloader, test_dataloader],
                                          [self.train_target_dir, self.val_target_dir, self.test_target_dir]):
            for batch in tqdm(dataloader):
                for input, data_sample in zip(batch['inputs'], batch['data_samples']):
                    save_path = os.path.join(self.proposals_cache_path, f"{data_sample.img_id}.pt")
                    if os.path.exists(save_path):
                        continue
                    img = input
                    heatmap = self.dino_vit_bg_subtractor.run_on_image_tensor(img)
                    padding_mask = create_padding_mask(image_path=data_sample.img_path, padding_value=[104, 116, 124])
                    predicted_patches, _, patches_scores_conv, patches_scores = (extract_patches_accord_heatmap
                                                                                 (heatmap=heatmap,
                                                                                  img_id=data_sample.img_id,
                                                                                  patch_size=self.proposals_sizes[
                                                                                      'square'],
                                                                                  plot=False,
                                                                                  threshold_percentage=
                                                                                  self.patches_filtering_threshold,
                                                                                  target_dir=target_dir,
                                                                                  image_path=data_sample.img_path,
                                                                                  padding_mask=padding_mask))
                    cache_dict = {}
                    cache_dict['predicted_patches'] = predicted_patches
                    cache_dict['patches_scores'] = patches_scores
                    cache_dict['patches_scores_conv'] = patches_scores_conv
                    try:
                        torch.save(cache_dict, save_path)
                    except Exception as e:
                        self.logger.error(f"Error saving cache file {e}")

    def initiate_dataloaders(self):
        self.anomaly_detector_cfg.train_dataloader.dataset.extracted_patches_folder = self.proposals_cache_path
        self.anomaly_detector_cfg.val_dataloader.dataset.extracted_patches_folder = self.proposals_cache_path
        self.anomaly_detector_cfg.test_dataloader.dataset.extracted_patches_folder = self.proposals_cache_path
        self.train_dataloader = create_dataloader(dataloader_cfg=self.anomaly_detector_cfg.train_dataloader)
        self.val_dataloader = create_dataloader(dataloader_cfg=self.anomaly_detector_cfg.val_dataloader)
        self.test_dataloader = create_dataloader(dataloader_cfg=self.anomaly_detector_cfg.test_dataloader)
        self.dataloaders = {"train": self.train_dataloader,
                            "val": self.val_dataloader,
                            "test": self.test_dataloader}
        self.classes_names = self.train_dataloader.dataset.METAINFO['classes']
        self.id_labels = [label_index for label_index, label in
                     enumerate(self.classes_names) if label not in self.train_dataloader.dataset.ood_labels]
        self.bbox_regressor_runner.visualizer.dataset_meta = self.train_dataloader.dataset.METAINFO


    def run(self):
        self.logger.info(f"Running anomaly detection stage\n")
        self.train_target_dir = os.path.join(self.dino_vit_bg_subtractor.target_dir, "train_dataset")
        os.makedirs(self.train_target_dir, exist_ok=True)
        self.val_target_dir = os.path.join(self.dino_vit_bg_subtractor.target_dir, "val_dataset")
        os.makedirs(self.val_target_dir, exist_ok=True)
        self.test_target_dir = os.path.join(self.dino_vit_bg_subtractor.target_dir, "test_dataset")
        os.makedirs(self.test_target_dir, exist_ok=True)
        self.logger.info(f"Anomaly detection - boxes regressor\n")
        if self.to_extract_patches:
            self.extract_patches()

        self.create_bbox_regressor_runner()
        self.bbox_regressor_runner.train()
        self.initiate_dataloaders()
        if not self.skip_stage:
            # self.logger.info(f"Anomaly detection - train dataset\n")
            # self.save_objects_for_dataset("train")
            #
            # self.logger.info(f"Anomaly detection - val dataset\n")
            # self.save_objects_for_dataset("val")

            self.logger.info(f"Anomaly detection - test dataset\n")
            self.save_objects_for_dataset("test")

            self.plot_ranks_graph_and_tt1_after_AD_stage()
        if self.evaluate_stage:
            self.test_with_and_without_regressor()

    def test_with_and_without_regressor(self):
        self.logger.info(f"Anomaly detection - Evaluate regressor\n")
        self.bbox_regressor_runner.test()
        # test without regressor
        predicted_bbox = []
        gts = []
        for batch in self.bbox_regressor_runner.test_dataloader:
            for data_inputs, data_sample in zip(batch['inputs'], batch['data_samples']):
                predicted_bbox.append(hbox2rbox(data_sample.predicted_patches.predicted_patches).cpu().numpy())
                gts.append({'bboxes': data_sample.gt_instances.bboxes.cpu().numpy(),
                            'bboxes_ignore': np.zeros((0, 5)),
                            'labels': data_sample.gt_instances.labels.cpu().numpy(),
                            'labels_ignore': np.array([])})

        scale_ranges = self.anomaly_detector_cfg.bbox_regressor.test_evaluator.scale_ranges
        classes = self.bbox_regressor_runner.test_dataloader.dataset.METAINFO['classes']
        eval_rbbox_mrecall_for_regressor(det_results=predicted_bbox, annotations=gts, scale_ranges=scale_ranges,
                                         dataset=classes, logger=self.logger)

    def plot_with_and_without_regressor(self):
        # test without regressor
        output_dir = os.path.join(self.bbox_regressor_output_dir, 'bbox_regressor_examples')
        os.makedirs(output_dir, exist_ok=True)
        visualizer = self.bbox_regressor_runner.visualizer
        for batch in self.bbox_regressor_runner.test_dataloader:
            for data_inputs, data_sample in zip(batch['inputs'], batch['data_samples']):
                img = cv2.imread(data_sample.img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # plot with regressor results
                data_sample.predicted_patches.predicted_patches = data_sample.predicted_patches.predicted_patches.to(
                    device)
                regressor_results = self.bbox_regressor_runner.model.predict(
                    data_inputs.unsqueeze(dim=0).to(device), [data_sample])
                results = regressor_results[0]
                results.pred_instances.scores = torch.tensor([1] * len(results.pred_instances.bboxes))
                results.pred_instances.labels = torch.tensor([0] * len(results.pred_instances.bboxes))
                visualizer.add_datasample(name='result',
                                          image=img,
                                          data_sample=results.detach().cpu(),
                                          draw_gt=True,
                                          out_file=os.path.join(output_dir,
                                                                data_sample.img_id + "_regressor_results.png"),
                                          wait_time=0,
                                          draw_text=False)
                # plot without regressor results
                results = InstanceData()
                results.bboxes = hbox2rbox(data_sample.predicted_patches.predicted_patches)
                results.scores = torch.tensor([1] * len(results.bboxes))
                results.labels = torch.tensor([0] * len(results.bboxes))
                data_sample.pred_instances = results
                visualizer.add_datasample(name='result',
                                          image=img,
                                          data_sample=data_sample.detach().cpu(),
                                          draw_gt=True,
                                          out_file=os.path.join(output_dir, data_sample.img_id + ".png"),
                                          wait_time=0,
                                          draw_text=False)

    def plot_ranks_graph_and_tt1_after_AD_stage(self):
        all_labels = []
        all_AD_socres = []
        for batch in tqdm(self.test_dataloader):
            for data_inputs, data_sample in zip(batch['inputs'], batch['data_samples']):
                formatted_proposals_patches = RotatedBoxes(hbox2rbox(data_sample.predicted_patches.predicted_patches))
                formatted_proposals_patches = InstanceData(priors=formatted_proposals_patches)
                assign_result = self.bbox_assigner.assign(
                    formatted_proposals_patches.to(device), data_sample.gt_instances.to(device))
                all_labels.extend(assign_result.labels.cpu().numpy())
                # open pt file a read for each patch its corresponds anomaly score
                curr_anomaly_scores_file = os.path.join(self.proposals_cache_path, f"{data_sample.img_id}.pt")
                with open(curr_anomaly_scores_file, 'rb') as f:
                    scores_dict = torch.load(f)
                all_AD_socres.extend(scores_dict['patches_scores'])

        all_AD_socres = torch.tensor(all_AD_socres)
        all_labels = torch.tensor(all_labels)
        sorted_all_anomaly_scores, sorted_all_anomaly_scores_indices = torch.sort(all_AD_socres)
        ood_labels = [self.test_dataloader.dataset.METAINFO['classes'].index(label) for label in
                      self.test_dataloader.dataset.ood_labels]
        tt1 = {}
        AD_ranks_dict = {}
        for OOD_label in ood_labels:
            if OOD_label not in all_labels:
                continue
            curr_label_anomaly_scores = all_AD_socres[all_labels == OOD_label]
            curr_label_sorted_anomaly_scores, curr_label_anomaly_scores_indices = torch.sort(
                curr_label_anomaly_scores)
            curr_label_ranks_in_all_anomaly_scores = len(
                sorted_all_anomaly_scores) - 1 - torch.searchsorted(sorted_all_anomaly_scores,
                                                                    curr_label_sorted_anomaly_scores)
            plt.plot(list(range(len(curr_label_ranks_in_all_anomaly_scores))),
                     torch.sort(curr_label_ranks_in_all_anomaly_scores)[0],
                     label=self.test_dataloader.dataset.METAINFO['classes'][OOD_label])
            tt1[self.test_dataloader.dataset.METAINFO['classes'][OOD_label]] = \
                torch.sort(curr_label_ranks_in_all_anomaly_scores)[0][0]
            self.logger.info(
                f"OOD label {self.test_dataloader.dataset.METAINFO['classes'][OOD_label]} first rank in AD"
                f" scores: {tt1[self.test_dataloader.dataset.METAINFO['classes'][OOD_label]]}")
            AD_ranks_dict[self.test_dataloader.dataset.METAINFO['classes'][OOD_label]] = list(
                torch.sort(curr_label_ranks_in_all_anomaly_scores)[0].numpy().astype(np.float64))

            # plt.plot(list(range(len(abnormal_ranks_in_sorted_anomaly_scores))), torch.sort(abnormal_ranks_in_sorted_anomaly_scores)[0])
        plt.xlabel(f'abnormal objects ranks in anomaly detection scores')
        plt.ylabel(f'all samples anomaly detection scores rank')
        plt.title('Abnormal objects ranks')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.test_output_dir, f"abnormal_ranks_in_anomaly_detection_scores.pdf"))

        sorted_data = sorted(tt1.items(), key=lambda x: x[1])
        classes_names, first_rank = zip(*sorted_data)
        fig = plt.figure(figsize=(10, 5))
        plt.bar(classes_names, first_rank, width=0.4)
        plt.xlabel("novel classes")
        plt.ylabel("TIme to first")
        plt.title("Time to first (TT-1) for novel classes")
        plt.grid(True)
        plt.yscale('log')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.test_output_dir, f"TT-1.pdf"))

        with open(os.path.join(self.test_output_dir, "AD_ranks_dict.json"), 'w') as f:
            json.dump(AD_ranks_dict, f, indent=4)

    def apply_nms_per_image(self, image_id, predicted_boxes, scores_dict, visualizer, iou_calculator, iou_threshold=0.5,
                            plot=False, dataset_type="train", dynamic_threshold=None, filter_thresh=None):
        target_dir = self.train_target_dir if dataset_type == "train" else self.val_target_dir if dataset_type == "val" else self.test_target_dir
        boxes_num_per_subimage = [pb.shape[0] for pb in predicted_boxes]
        predicted_boxes = torch.concat(predicted_boxes, dim=0)
        # Sort boxes by scores in descending order
        scores = torch.cat([torch.tensor(sub_image_scores["patches_scores"]) for sub_image_scores in scores_dict])
        indices = scores.clone().argsort(descending=True)
        keep = []
        if dynamic_threshold:
            path = os.path.join(self.data_path, "DOTAV2_ss/test/images/")
            filter_thresh = len([f for f in os.listdir(path) if f.startswith(image_id)])*100

        while len(indices) > 0:
            current = indices[0]
            keep.append(current.item())
            if len(indices) == 1:
                break

            current_box = predicted_boxes[current].unsqueeze(0)
            remaining_boxes = predicted_boxes[indices[1:]]

            ious = iou_calculator(current_box, remaining_boxes).squeeze(0)

            # Keep boxes with IoU less than the threshold
            indices = indices[1:][ious <= iou_threshold]
        # filter keep according to filter_thresh
        keep = keep[:filter_thresh] if filter_thresh else keep
        img_patches = predicted_boxes[keep]
        scores = [scores[ind] for ind in keep]
        accum_boxes_num_per_subimage = list(itertools.accumulate(boxes_num_per_subimage))
        keep_sorted = sorted(keep)
        keep_indices_per_subimage = [[] for _ in range(len(boxes_num_per_subimage))]
        curr_sub_img=0
        accumulated_boxes_num = 0
        for ind in keep_sorted:
            while ind >= accum_boxes_num_per_subimage[curr_sub_img]:
                curr_sub_img+=1
                accumulated_boxes_num+=boxes_num_per_subimage[curr_sub_img-1]
            keep_indices_per_subimage[curr_sub_img].append(ind-accumulated_boxes_num)

        if plot:
            metadata_path = os.path.join(self.original_data_path, dataset_type, "meta", f"{image_id}.txt")
            with open(metadata_path, 'r') as f:
                for line in f:
                    if line.startswith('gsd'):
                        num = line.split(':')[-1]
                        try:
                            orig_gsd = float(num)
                        except ValueError:
                            orig_gsd = None
            gsds_div = self.lowest_gsd / orig_gsd
            labels_path=os.path.join(self.original_data_path, dataset_type, "labelTxt", f"{image_id}.txt")
            objects = parse_dota_poly(labels_path)
            gt_labels = [self.classes_names.index(obj["name"]) for obj in objects]
            formatted_objects = qbox2rbox(torch.stack([torch.tensor(obj["poly"]).flatten()/gsds_div for obj in objects]))
            img_path=os.path.join(self.original_data_path, dataset_type, "images", f"{image_id}.png")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            width, height = Image.open(img_path).size
            img = cv2.resize(img, (ceil(width / gsds_div), ceil(height / gsds_div)))
            # fix predicted boxes to lowest gsd
            results = DetDataSample()
            results.gt_instances, results.pred_instances = InstanceData(), InstanceData()
            results.gt_instances.bboxes = formatted_objects
            results.gt_instances.labels = torch.tensor(gt_labels)
            results.pred_instances.bboxes = img_patches
            results.pred_instances.scores = torch.tensor([1] * len(results.pred_instances.bboxes))
            results.pred_instances.labels = torch.tensor([0] * len(results.pred_instances.bboxes))
            visualizer.add_datasample(name=f"{image_id}_post_nms_multiscale_results",
                                      image=img,
                                      data_sample=results.detach().cpu(),
                                      draw_gt=True,
                                      out_file=os.path.join(target_dir,
                                                             f"{image_id}_post_nms_multiscale_results.png"),
                                      wait_time=0,
                                      draw_text=False)
        return keep_indices_per_subimage

    def apply_nms_and_save_objects(self, prev_image_id, curr_all_boxes, curr_all_scores_dict,
                                   curr_regressor_results,
                                   hashmap_locations_and_anomaly_scores, dataset_type, plot=False,
                                   dynamic_threshold=None, filter_thresh=None):
        target_dir = self.train_target_dir if dataset_type == "train" else self.val_target_dir if dataset_type == "val" else self.test_target_dir
        output_dir = self.output_dir_train_dataset if dataset_type == "train" else self.output_dir_val_dataset if dataset_type == "val" else self.output_dir_test_dataset
        keep_indices_per_subimage = self.apply_nms_per_image(image_id=prev_image_id, predicted_boxes=curr_all_boxes,
                                                             scores_dict=curr_all_scores_dict,
                                                             visualizer=self.bbox_regressor_runner.visualizer,
                                                             iou_calculator=self.bbox_assigner.iou_calculator, plot=plot,
                                                             dataset_type=dataset_type, dynamic_threshold=dynamic_threshold, filter_thresh=filter_thresh)
        # save patches per subimage
        for sub_image_ind, (regressor_result, keep_inds) in enumerate(zip(curr_regressor_results, keep_indices_per_subimage)):
            if len(keep_inds) == 0:
                continue
            regressor_result.pred_instances = regressor_result.pred_instances[keep_inds]
            filtered_scored_dict = {'patches_scores': [curr_all_scores_dict[sub_image_ind]['patches_scores'][ind] for ind in keep_inds],
                                    'patches_scores_conv': [curr_all_scores_dict[sub_image_ind]['patches_scores_conv'][ind] for ind in keep_inds]}
            assign_predicted_boxes_to_gt_boxes_and_save(bbox_assigner=self.bbox_assigner,
                                                        regressor_results=regressor_result,
                                                        image_path=regressor_result.img_path,
                                                        img_id=regressor_result.img_id,
                                                        labels_names=self.classes_names,
                                                        logger=self.logger,
                                                        plot=False,
                                                        dataset_type=dataset_type,
                                                        target_dir=target_dir,
                                                        extract_bbox_path=output_dir,
                                                        visualizer=self.bbox_regressor_runner.visualizer,
                                                        hashmap_locations=hashmap_locations_and_anomaly_scores,
                                                        scores_dict=filtered_scored_dict
                                                        )

    def save_objects_for_dataset(self, dataset_type:Union["train", "val", "test"]):
        hashmap_locations_and_anomaly_scores_file = self.hashmap_locations_and_anomaly_scores_train_file if dataset_type == "train" else \
            self.hashmap_locations_and_anomaly_scores_val_file if dataset_type == "val" else \
            self.hashmap_locations_and_anomaly_scores_test_file
        output_dir = self.output_dir_train_dataset if dataset_type == "train" else \
            self.output_dir_val_dataset if dataset_type == "val" else \
            self.output_dir_test_dataset
        dynamic_threshold = None
        filter_thresh = None
        # if dataset_type == "test":
        #     dynamic_threshold=False
            # filter_thresh=500
        dataloader = self.dataloaders[dataset_type]
        if os.path.exists(hashmap_locations_and_anomaly_scores_file):
            os.remove(hashmap_locations_and_anomaly_scores_file)
        hashmap_locations_and_anomaly_scores = {}
        curr_all_boxes = []
        curr_regressor_results = []
        curr_all_scores_dict = []
        prev_image_id = None
        # plot 10 first images
        plot=True
        i=0
        with torch.no_grad():
            for batch in tqdm(dataloader):
                for data_inputs, data_sample in zip(batch['inputs'], batch['data_samples']):
                    data_sample = data_sample.to(device)
                    data_inputs = data_inputs.to(device)
                    curr_image_id = data_sample.img_id[:data_sample.img_id.find("__")]
                    if prev_image_id is None:
                        prev_image_id = curr_image_id
                    if curr_image_id != prev_image_id:
                        self.apply_nms_and_save_objects(prev_image_id=prev_image_id, curr_all_boxes=curr_all_boxes,
                                                        curr_all_scores_dict = curr_all_scores_dict,
                                                        curr_regressor_results=curr_regressor_results,
                                                        hashmap_locations_and_anomaly_scores=hashmap_locations_and_anomaly_scores,
                                                        dataset_type=dataset_type, plot=plot, dynamic_threshold=dynamic_threshold,
                                                        filter_thresh=filter_thresh)
                        i+=1
                        if i>10:
                            plot=False
                        curr_all_boxes = []
                        curr_regressor_results = []
                        curr_all_scores_dict = []
                        prev_image_id = curr_image_id
                    # inject ID gts in train dataset
                    if dataset_type=="train":
                        save_id_gts(gt_instances=data_sample.gt_instances, image_path=data_sample.img_path,
                                    id_classes_names=[self.classes_names[i] for i in self.id_labels],
                                    extract_bbox_path=output_dir,
                                    id_class_labels=self.id_labels, logger=self.logger, img_id=data_sample.img_id)
                    # iterate per image to save all patches
                    if data_sample.predicted_patches.predicted_patches.shape[0]==0:
                        continue
                    torch.cuda.empty_cache()
                    curr_regressor_result = self.bbox_regressor_runner.model.predict(
                        data_inputs.unsqueeze(dim=0), [data_sample])[0].to(device)
                    curr_anomaly_scores_file = os.path.join(self.proposals_cache_path, f"{data_sample.img_id}.pt")
                    with open(curr_anomaly_scores_file, 'rb') as f:
                        scores_dict = torch.load(f)
                        del scores_dict['predicted_patches']
                    curr_all_boxes.append(self.adapt_patches(curr_regressor_result.pred_instances.bboxes, data_sample.img_id))
                    curr_all_scores_dict.append(scores_dict)
                    curr_regressor_results.append(curr_regressor_result)

            assert curr_image_id == prev_image_id  # last image
            self.apply_nms_and_save_objects(prev_image_id=prev_image_id, curr_all_boxes=curr_all_boxes,
                                            curr_all_scores_dict=curr_all_scores_dict,
                                            curr_regressor_results=curr_regressor_results,
                                            hashmap_locations_and_anomaly_scores=hashmap_locations_and_anomaly_scores,
                                            dataset_type=dataset_type, plot=plot)
            with open(hashmap_locations_and_anomaly_scores_file, 'w') as f:
                json.dump(hashmap_locations_and_anomaly_scores, f, indent=4)


    def adapt_patches(self, patches, img_id):
        orig_gsd = os.path.splitext(img_id)[0].split("_")[-1]
        orig_gsd = float(orig_gsd[0] + "." + orig_gsd[1:])
        shift_x = int(img_id.split("__")[2])
        shift_y = int(img_id.split("___")[1].split("_")[0])
        gsds_div = self.lowest_gsd / orig_gsd
        adapted_patches = patches.clone().to(device)
        adapted_patches[:, 0] = adapted_patches[:, 0] + shift_x
        adapted_patches[:, 1] = adapted_patches[:, 1] + shift_y
        adapted_patches[:, :-1]= torch.ceil(adapted_patches[:, :-1]/gsds_div).int()
        return adapted_patches

