
import os

import cv2

import random
from mmrotate.structures.bbox import hbox2rbox
from mmrotate.evaluation import eval_rbbox_mrecall_for_regressor
import numpy as np
import torch
from bbox_regressor import BBoxRegressor
from mmengine.runner import Runner
from tqdm import tqdm
from single_image_bg_detector.bg_subtraction_with_dino_vit import BGSubtractionWithDinoVit
from DOTA_devkit.DOTA import DOTA
from single_image_bg_detector.bg_subtractor_utils import (extract_patches_accord_heatmap, \
                                                          assign_predicted_boxes_to_gt_boxes_and_save_val_stage, \
                                                          assign_predicted_boxes_to_gt_boxes_and_save)
from utils import create_dataloader
from mmdet.registry import TASK_UTILS
from mmengine.structures import InstanceData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
from mmdet.registry import MODELS, TASK_UTILS


class AnomalyDetector:
    """Abstract class of anomaly detector. It receives output dir path, dataset configuration and embedder configuration"""

    def __init__(self, output_dir, anomaly_detector_cfg, logger, current_run_name):
        self.anomaly_detector_cfg = anomaly_detector_cfg
        self.logger = logger
        self.output_dir = os.path.join(output_dir, "train/anomaly_detection_result", current_run_name)
        self.data_output_dir = os.path.join(self.output_dir,
                                            anomaly_detector_cfg.data_output_dir_name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_output_dir, exist_ok=True)
        # create train, test and validation datasets
        self.proposals_cache_path = os.path.join(self.output_dir, 'cache_folder')
        os.makedirs(self.proposals_cache_path, exist_ok=True)

        self.output_dir_train_dataset = os.path.join(self.data_output_dir, "train")
        self.output_dir_val_dataset = os.path.join(self.data_output_dir, "val")
        self.output_dir_test_dataset = os.path.join(self.data_output_dir, "test")
        self.to_extract_patches = anomaly_detector_cfg.extract_patches

    def run(self):
        pass


class VitBasedAnomalyDetector(AnomalyDetector):
    # extract patches and save them according to their assigned labels
    def __init__(self, anomaly_detector_cfg, output_dir, logger, current_run_name):
        super().__init__(output_dir=output_dir, anomaly_detector_cfg=anomaly_detector_cfg, logger=logger, current_run_name=current_run_name)
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

        self.skip_stage = anomaly_detector_cfg.skip_stage
        self.bbox_regressor_output_dir = os.path.join(self.output_dir, 'bbox_regressor')
        anomaly_detector_cfg.bbox_regressor['work_dir'] = self.bbox_regressor_output_dir

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
        # visualizer = self.bbox_regressor_runner.visualizer
        # img = self.bbox_regressor_runner.test_dataloader.dataset[0]['inputs'].cpu().numpy().transpose((1, 2, 0))
        # img = mmcv.imconvert(img, 'bgr', 'rgb')
        # visualizer.add_datasample(name='result',
        #         image=img,
        #         data_sample=self.bbox_regressor_runner.test_dataloader.dataset[0]['data_samples'],
        #         draw_gt=True,
        #         show=True,
        #         wait_time=0)
        # x=1

    def extract_patches(self):
        self.logger.info(f"Extract patches\n")
        train_dataloader = create_dataloader(dataloader_cfg=self.anomaly_detector_cfg.bbox_regressor.train_dataloader)
        val_dataloader = create_dataloader(dataloader_cfg=self.anomaly_detector_cfg.bbox_regressor.val_dataloader)
        test_dataloader = create_dataloader(dataloader_cfg=self.anomaly_detector_cfg.bbox_regressor.test_dataloader)
        for dataloader in [train_dataloader, val_dataloader, test_dataloader]:
            for batch in tqdm(dataloader):
                for input, data_sample in zip(batch['inputs'], batch['data_samples']):
                    save_path = os.path.join(self.proposals_cache_path, f"{data_sample.img_id}.pt")
                    if os.path.exists(save_path):
                        continue
                    img = input
                    heatmap = self.dino_vit_bg_subtractor.run_on_image_tensor(img)
                    predicted_patches, _, _ = extract_patches_accord_heatmap(heatmap=heatmap,
                                                                             img_id=data_sample.img_id,
                                                                             patch_size=self.proposals_sizes['square'],
                                                                             plot=False,
                                                                             threshold_percentage=
                                                                             self.patches_filtering_threshold)
                    cache_dict = {}
                    cache_dict['predicted_patches'] = predicted_patches
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


    def run(self):
        self.logger.info(f"Running anomaly detection stage\n")
        if not self.skip_stage:
            train_target_dir = os.path.join(self.dino_vit_bg_subtractor.target_dir, "train_dataset")
            os.makedirs(train_target_dir, exist_ok=True)
            val_target_dir = os.path.join(self.dino_vit_bg_subtractor.target_dir, "val_dataset")
            os.makedirs(val_target_dir, exist_ok=True)
            test_target_dir = os.path.join(self.dino_vit_bg_subtractor.target_dir, "test_dataset")
            os.makedirs(test_target_dir, exist_ok=True)
            self.logger.info(f"Anomaly detection - boxes regressor\n")
            if self.to_extract_patches:
                self.extract_patches()
            self.create_bbox_regressor_runner()
            self.bbox_regressor_runner.train()
            # self.test_with_and_without_regressor()
            self.initiate_dataloaders()
            self.logger.info(f"Anomaly detection - train dataset\n")
            self.bbox_regressor_runner.visualizer.dataset_meta = self.train_dataloader.dataset.METAINFO
            for batch in tqdm(self.train_dataloader):
                for data_inputs, data_sample in zip(batch['inputs'], batch['data_samples']):
                    # iterate per image to save all patches
                    data_sample.predicted_patches.predicted_patches = data_sample.predicted_patches.predicted_patches.to(device)
                    regressor_results = self.bbox_regressor_runner.model.predict(
                        data_inputs.unsqueeze(dim=0).to(device), [data_sample])
                    assign_predicted_boxes_to_gt_boxes_and_save(bbox_assigner=self.bbox_assigner,
                                                                regressor_results=regressor_results,
                                                                gt_instances=data_sample.gt_instances,
                                                                image_path=data_sample.img_path,
                                                                img_id=data_sample.img_id,
                                                                labels_names=self.classes_names,
                                                                logger=self.logger,
                                                                plot=False,
                                                                target_dir=train_target_dir,
                                                                extract_bbox_path=self.output_dir_train_dataset,
                                                                visualizer=self.bbox_regressor_runner.visualizer)
            self.logger.info(f"Anomaly detection - val dataset\n")
            for batch in tqdm(self.val_dataloader):
                for data_inputs, data_sample in zip(batch['inputs'], batch['data_samples']):
                    # iterate per image to save all patches
                    data_sample.predicted_patches.predicted_patches = data_sample.predicted_patches.predicted_patches.to(device)
                    regressor_results = self.bbox_regressor_runner.model.predict(
                        data_inputs.unsqueeze(dim=0).to(device), [data_sample])
                    assign_predicted_boxes_to_gt_boxes_and_save(bbox_assigner=self.bbox_assigner,
                                                                regressor_results=regressor_results,
                                                                gt_instances=data_sample.gt_instances,
                                                                image_path=data_sample.img_path,
                                                                img_id=data_sample.img_id,
                                                                labels_names=self.classes_names,
                                                                logger=self.logger,
                                                                plot=False,
                                                                target_dir=val_target_dir,
                                                                extract_bbox_path=self.output_dir_val_dataset,
                                                                visualizer=self.bbox_regressor_runner.visualizer,
                                                                train=False, val=True)

            self.logger.info(f"Anomaly detection - test dataset\n")
            for batch in tqdm(self.test_dataloader):
                for data_inputs, data_sample in zip(batch['inputs'], batch['data_samples']):
                    # iterate per image to save all patches
                    data_sample.predicted_patches.predicted_patches = data_sample.predicted_patches.predicted_patches.to(device)
                    regressor_results = self.bbox_regressor_runner.model.predict(
                        data_inputs.unsqueeze(dim=0).to(device), [data_sample])
                    assign_predicted_boxes_to_gt_boxes_and_save(bbox_assigner=self.bbox_assigner,
                                                                regressor_results=regressor_results,
                                                                gt_instances=data_sample.gt_instances,
                                                                image_path=data_sample.img_path,
                                                                img_id=data_sample.img_id,
                                                                labels_names=self.classes_names,
                                                                logger=self.logger,
                                                                plot=False,
                                                                target_dir=test_target_dir,
                                                                extract_bbox_path=self.output_dir_test_dataset,
                                                                visualizer=self.bbox_regressor_runner.visualizer,
                                                                train=False)

    def test_with_and_without_regressor(self):
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
