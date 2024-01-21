import json
import os

import random
import numpy as np
import torch
from tqdm import tqdm
from single_image_bg_detector.bg_subtraction_with_dino_vit import BGSubtractionWithDinoVit
from DOTA_devkit.DOTA import DOTA
from results import analyze_roc_curve, plot_graphs
from single_image_bg_detector.bg_subtractor_utils import extract_patches_accord_heatmap
from utils import create_dataloader
from mmdet.registry import TASK_UTILS

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
    def __init__(self, output_dir, anomaly_detector_cfg, logger):
        self.logger = logger
        self.output_dir = os.path.join(self.output_dir, "train/anomaly_detection_result")
        os.makedirs(self.output_dir, exist_ok=True)
        # create train, test and validation datasets
        self.train_dataloader = create_dataloader(cfg=anomaly_detector_cfg.train_dataloader)
        self.val_data_loader = create_dataloader(cfg=anomaly_detector_cfg.val_dataloader)
        self.dataloaders = {"train": self.train_dataloader,
                            "val": self.val_data_loader}

    def run(self):
        pass


class VitBasedAnomalyDetector(AnomalyDetector):
    # extract patches and save them according to their assigned labels
    def __init__(self, anomaly_detetctor_cfg, output_dir, logger):
        super().__init__(output_dir=output_dir, anomaly_detector_cfg=anomaly_detetctor_cfg, logger=logger)
        self.dota_obj = DOTA(basepath=anomaly_detetctor_cfg.train_dataloader.dataset.data_root)
        self.dino_vit_bg_subtractor = BGSubtractionWithDinoVit(target_dir=self.output_dir,
                                                          vit_patch_size=anomaly_detetctor_cfg.vit_patch_size,
                                                          vit_arch=anomaly_detetctor_cfg.vit_arch,
                                                          vit_image_size=anomaly_detetctor_cfg.vit_image_size,
                                                          dota_obj=self.dota_obj,
                                                          threshold=anomaly_detetctor_cfg.vit_threshold,
                                                          pretrained_weights=anomaly_detetctor_cfg.pretrained_weights,
                                                          checkpoint_key=anomaly_detetctor_cfg.checkpoint_key,
                                                          model_mode=anomaly_detetctor_cfg.vit_model_mode,
                                                          model_type=anomaly_detetctor_cfg.vit_model_type)
        self.bbox_assigner = TASK_UTILS.build(anomaly_detetctor_cfg.patches_assigner)
        self.proposals_sizes=anomaly_detetctor_cfg.proposals_sizes
        self.patches_filtering_threshold = anomaly_detetctor_cfg.patches_filtering_threshold

    def run(self):
        for batch in tqdm(self.train_dataloader):
            # measure performance for images with objects only
            if batch['data_samples'][0].gt_instances.bboxes.size()[0] > 0:
                heatmap = self.dino_vit_bg_subtractor.run_on_image_tensor(img=batch['inputs'][0])
                predicted_patches, mask, scores = extract_patches_accord_heatmap(heatmap=heatmap,
                                                                                 img_id=batch['data_samples'][0].img_id,
                                                                                 patch_size=self.proposals_sizes['square'],
                                                                                 plot=False,
                                                                                 threshold_percentage=self.patches_filtering_threshold)
                gt_labels, gt_inds, dt_labels, dt_match = assign_predicted_boxes_to_gt(bbox_assigner=bbox_assigner,
                                                                                       predicted_boxes=predicted_patches,
                                                                                       data_batch=batch,
                                                                                       all_labels=all_labels,
                                                                                       img_id=batch['data_samples'][
                                                                                           0].img_id,
                                                                                       dota_obj=dota_obj,
                                                                                       heatmap=heatmap,
                                                                                       patch_size=proposal_bbox_size,
                                                                                       plot=plot, title=title)

