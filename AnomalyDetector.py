import json
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from single_image_bg_detector.bg_subtraction_with_dino_vit import BGSubtractionWithDinoVit
from DOTA_devkit.DOTA import DOTA
from single_image_bg_detector.bg_subtractor_utils import extract_patches_accord_heatmap, \
    assign_predicted_boxes_to_gt_boxes_using_hypothesis, assign_predicted_boxes_to_gt_boxes_and_save_val_stage
from utils import create_dataloader
from mmdet.registry import TASK_UTILS

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


class AnomalyDetector:
    """Abstract class of anomaly detector. It receives output dir path, dataset configuration and embedder configuration"""

    def __init__(self, output_dir, anomaly_detector_cfg, logger):
        self.logger = logger
        self.output_dir = os.path.join(output_dir, "train/anomaly_detection_result")
        self.data_output_dir = os.path.join(self.output_dir, anomaly_detector_cfg.data_output_dir_name)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_output_dir, exist_ok=True)
        # create train, test and validation datasets
        self.train_dataloader = create_dataloader(dataloader_cfg=anomaly_detector_cfg.train_dataloader)
        self.val_dataloader = create_dataloader(dataloader_cfg=anomaly_detector_cfg.val_dataloader)
        self.test_dataloader = create_dataloader(dataloader_cfg=anomaly_detector_cfg.test_dataloader)
        self.dataloaders = {"train": self.train_dataloader,
                            "val": self.val_dataloader,
                            "test": self.test_dataloader}
        self.output_dir_train_dataset = os.path.join(self.data_output_dir, "train")
        self.output_dir_val_dataset = os.path.join(self.data_output_dir, "val")
        self.output_dir_test_dataset = os.path.join(self.data_output_dir, "test")

    def run(self):
        pass


class VitBasedAnomalyDetector(AnomalyDetector):
    # extract patches and save them according to their assigned labels
    def __init__(self, anomaly_detetctor_cfg, output_dir, logger):
        super().__init__(output_dir=output_dir, anomaly_detector_cfg=anomaly_detetctor_cfg, logger=logger)
        self.logger.info(f"Creating anomaly detector\n")
        self.dota_obj_train = DOTA(basepath=anomaly_detetctor_cfg.train_dataloader.dataset.data_root)
        self.dota_obj_val = DOTA(basepath=anomaly_detetctor_cfg.val_dataloader.dataset.data_root)
        self.dota_obj_test = DOTA(basepath=anomaly_detetctor_cfg.test_dataloader.dataset.data_root)
        self.dino_vit_bg_subtractor = BGSubtractionWithDinoVit(target_dir=self.output_dir,
                                                               vit_patch_size=anomaly_detetctor_cfg.vit_patch_size,
                                                               vit_arch=anomaly_detetctor_cfg.vit_arch,
                                                               vit_image_size=anomaly_detetctor_cfg.vit_image_size,
                                                               dota_obj=self.dota_obj_train,
                                                               threshold=anomaly_detetctor_cfg.vit_threshold,
                                                               pretrained_weights=anomaly_detetctor_cfg.pretrained_weights,
                                                               checkpoint_key=anomaly_detetctor_cfg.checkpoint_key,
                                                               model_mode=anomaly_detetctor_cfg.vit_model_mode,
                                                               model_type=anomaly_detetctor_cfg.vit_model_type)
        self.bbox_assigner = TASK_UTILS.build(anomaly_detetctor_cfg.patches_assigner)
        self.proposals_sizes = anomaly_detetctor_cfg.proposals_sizes
        self.patches_filtering_threshold = anomaly_detetctor_cfg.patches_filtering_threshold
        self.classes_names = self.train_dataloader.dataset.METAINFO['classes']
        self.skip_stage = anomaly_detetctor_cfg.skip_stage

    def run(self):
        self.logger.info(f"Running anomaly detection stage\n")
        if not self.skip_stage:
            train_target_dir = os.path.join(self.dino_vit_bg_subtractor.target_dir, "train_dataset")
            os.makedirs(train_target_dir, exist_ok=True)
            val_target_dir = os.path.join(self.dino_vit_bg_subtractor.target_dir, "val_dataset")
            os.makedirs(val_target_dir, exist_ok=True)
            test_target_dir = os.path.join(self.dino_vit_bg_subtractor.target_dir, "test_dataset")
            os.makedirs(test_target_dir, exist_ok=True)
            self.logger.info(f"Anomaly detection - train dataset\n")
            for batch in tqdm(self.train_dataloader):
                # measure performance for images with objects only
                if batch['data_samples'][0].gt_instances.bboxes.size()[0] > 0:
                    heatmap = self.dino_vit_bg_subtractor.run_on_image_tensor(img=batch['inputs'][0])

                    predicted_patches, _, _ = extract_patches_accord_heatmap(heatmap=heatmap,
                                                                             img_id=batch['data_samples'][0].img_id,
                                                                             patch_size=self.proposals_sizes['square'],
                                                                             plot=False,
                                                                             threshold_percentage=self.patches_filtering_threshold,
                                                                             target_dir=train_target_dir)

                    assign_predicted_boxes_to_gt_boxes_using_hypothesis(bbox_assigner=self.bbox_assigner,
                                                                        predicted_boxes=predicted_patches,
                                                                        data_batch=batch,
                                                                        img_id=batch['data_samples'][
                                                                            0].img_id,
                                                                        dota_obj=self.dota_obj_train,
                                                                        heatmap=heatmap,
                                                                        labels_names=self.classes_names,
                                                                        logger=self.logger,
                                                                        patch_size=self.proposals_sizes["square"],
                                                                        plot=False,
                                                                        target_dir=train_target_dir,
                                                                        extract_bbox_path=self.output_dir_train_dataset)
            self.logger.info(f"Anomaly detection - val dataset\n")
            for batch in tqdm(self.val_dataloader):
                heatmap = self.dino_vit_bg_subtractor.run_on_image_tensor(img=batch['inputs'][0])
                predicted_patches, _, _ = extract_patches_accord_heatmap(heatmap=heatmap,
                                                                         img_id=batch['data_samples'][0].img_id,
                                                                         patch_size=self.proposals_sizes['square'],
                                                                         plot=False,
                                                                         threshold_percentage=self.patches_filtering_threshold,
                                                                         target_dir=val_target_dir)
                assign_predicted_boxes_to_gt_boxes_and_save_val_stage(bbox_assigner=self.bbox_assigner,
                                                                      predicted_boxes=predicted_patches,
                                                                      data_batch=batch,
                                                                      img_id=batch['data_samples'][0].img_id,
                                                                      dota_obj=self.dota_obj_val,
                                                                      heatmap=heatmap,
                                                                      labels_names=self.classes_names,
                                                                      patch_size=self.proposals_sizes["square"],
                                                                      logger=self.logger,
                                                                      plot=False,
                                                                      target_dir=val_target_dir,
                                                                      extract_bbox_path=self.output_dir_val_dataset,
                                                                      is_val=True)

            self.logger.info(f"Anomaly detection - test dataset\n")
            for batch in tqdm(self.test_dataloader):
                heatmap = self.dino_vit_bg_subtractor.run_on_image_tensor(img=batch['inputs'][0])
                predicted_patches, _, _ = extract_patches_accord_heatmap(heatmap=heatmap,
                                                                         img_id=batch['data_samples'][0].img_id,
                                                                         patch_size=self.proposals_sizes['square'],
                                                                         plot=False,
                                                                         threshold_percentage=self.patches_filtering_threshold,
                                                                         target_dir=test_target_dir)
                assign_predicted_boxes_to_gt_boxes_and_save_val_stage(bbox_assigner=self.bbox_assigner,
                                                                      predicted_boxes=predicted_patches,
                                                                      data_batch=batch,
                                                                      img_id=batch['data_samples'][0].img_id,
                                                                      dota_obj=self.dota_obj_test,
                                                                      heatmap=heatmap,
                                                                      labels_names=self.classes_names,
                                                                      patch_size=self.proposals_sizes["square"],
                                                                      logger=self.logger,
                                                                      plot=False,
                                                                      target_dir=test_target_dir,
                                                                      extract_bbox_path=self.output_dir_test_dataset)
