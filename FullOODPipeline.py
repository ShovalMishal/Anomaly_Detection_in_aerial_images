import json
import numpy as np
import os
from argparse import ArgumentParser
import torch
from mmengine.config import Config

from PrepareDataPyramid import PrepareDataPyramid

from AnomalyDetector import VitBasedAnomalyDetector
from Classifier import VitClassifier, ResNet50Classifier, ResNet18Classifier
from OODDetector import ODINOODDetector, EnergyOODDetector, ViMOODDetector, MSPOODDetector
from utils import create_logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.multiprocessing.set_sharing_strategy('file_system')

# Set random seed for PyTorch
torch.manual_seed(1234)

# Set random seed for NumPy
np.random.seed(1234)

class FullODDPipeline:
    def __init__(self, args):
        self.cfg = Config.fromfile(args.config)
        self.output_dir = self.cfg.output_dir
        os.makedirs(os.path.join(self.output_dir), exist_ok=True)
        self.current_run_name = self.cfg.get("current_run_name")
        prepare_data_pyramid_cfg = self.cfg.get("prepare_data_pyramid_cfg")
        pdp = PrepareDataPyramid(prepare_data_pyramid_cfg)
        if not pdp.skip_stage:
            pdp.prepare_data_pyramid()
        anomaly_detector_cfg = self.cfg.get("anomaly_detector_cfg")
        self.classifier_cfg = self.cfg.get("classifier_cfg")
        self.OOD_detector_cfg = self.cfg.get("OOD_detector_cfg")
        self.logger = create_logger(os.path.join(self.output_dir,
                                                 f"{self.current_run_name}_full_ood_pipeline_log.log"))
        self.anomaly_detector = {'vit_based_anomaly_detector': VitBasedAnomalyDetector}[anomaly_detector_cfg.type]
        self.anomaly_detector = self.anomaly_detector(anomaly_detector_cfg,
                                                      self.output_dir, self.logger, self.current_run_name,
                                                      original_data_path=pdp.get_original_data_path(),
                                                      lowest_gsd=pdp.get_lowest_gsd())
        self.classifier = {'vit': VitClassifier, 'resnet50': ResNet50Classifier, 'resnet18':ResNet18Classifier}[self.classifier_cfg.type]
        self.classifier = self.classifier(self.output_dir, self.classifier_cfg, self.logger, self.current_run_name,
                                          self.anomaly_detector.data_output_dir,
                                          self.OOD_detector_cfg.ood_class_names)
        self.OOD_detector = {'ODIN': ODINOODDetector, 'Energy': EnergyOODDetector, 'msp': MSPOODDetector,
                             'vim': ViMOODDetector}[self.OOD_detector_cfg.type]
        self.OOD_detector = self.OOD_detector(cfg=self.OOD_detector_cfg, output_dir=self.output_dir,
                                              dataset_dir=self.anomaly_detector.data_output_dir,
                                              hashmap_locations_and_anomaly_scores_test_file=self.anomaly_detector.hashmap_locations_and_anomaly_scores_test_file,
                                              original_data_path=self.anomaly_detector.data_path,
                                              current_run_name=self.current_run_name,
                                              logger=self.logger)

    def train(self):
        self.logger.info("Starting training stage...")
        os.makedirs(os.path.join(self.output_dir, "train"), exist_ok=True)
        self.anomaly_detector.run()
        self.classifier.train()

    def test(self):
        self.logger.info("Starting testing stage...")
        os.makedirs(os.path.join(self.output_dir, "test"), exist_ok=True)
        model = self.classifier.get_fine_tuned_model().to(device).eval()
        self.classifier.evaluate_classifier(best_model=model)
        self.OOD_detector.test(model=model, train_transforms=self.classifier.train_transforms,
                               val_transforms=self.classifier.val_transforms,
                               visualizer=self.anomaly_detector.bbox_regressor_runner.visualizer,
                               dota_test_dataset=self.anomaly_detector.test_dataloader.dataset)


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="The relative path to the cfg file")
    args = parser.parse_args()
    pipeline = FullODDPipeline(args)
    pipeline.train()
    pipeline.test()


if __name__ == '__main__':
    main()
