import json
import numpy as np
import os
from argparse import ArgumentParser
import torch
from mmengine.config import Config

from AnomalyDetector import VitBasedAnomalyDetector, retrieve_anomaly_scores_for_test_dataset
from Classifier import VitClassifier, ResNet18Classifier
from OODDetector import ODINOODDetector, EnergyOODDetector, ViMOODDetector, save_k_outliers, \
    rank_samples_accord_features, MSPOODDetector, save_TT_1_images
from Classifier import create_dataloaders, DatasetType
from results import plot_graphs
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
        anomaly_detector_cfg = self.cfg.get("anomaly_detector_cfg")
        self.classifier_cfg = self.cfg.get("classifier_cfg")
        self.OOD_detector_cfg = self.cfg.get("OOD_detector_cfg")
        self.logger = create_logger(os.path.join(self.output_dir,
                                                 f"{self.current_run_name}_full_ood_pipeline_log.log"))
        self.anomaly_detector = {'vit_based_anomaly_detector': VitBasedAnomalyDetector}[anomaly_detector_cfg.type]
        self.anomaly_detector = self.anomaly_detector(anomaly_detector_cfg,
                                                      self.output_dir, self.logger, self.current_run_name)
        self.classifier = {'vit': VitClassifier, 'resnet18': ResNet18Classifier}[self.classifier_cfg.type]
        self.classifier = self.classifier(self.output_dir, self.classifier_cfg, self.logger, self.current_run_name)
        self.OOD_detector = {'ODIN': ODINOODDetector, 'Energy': EnergyOODDetector, 'msp': MSPOODDetector,
                             'vim': ViMOODDetector}[self.OOD_detector_cfg.type]
        self.OOD_detector = self.OOD_detector(output_dir=self.output_dir, current_run_name=self.current_run_name)

    def train(self):
        os.makedirs(os.path.join(self.output_dir, "train"), exist_ok=True)
        self.anomaly_detector.run()
        self.classifier.initiate_trainer(data_source=self.anomaly_detector.data_output_dir,
                                         ood_class_names=self.OOD_detector_cfg.ood_class_names)
        if self.classifier_cfg.retrain:
            self.classifier.train()

    def test(self):
        torch.cuda.empty_cache()
        self.logger.info("Starting testing stage...")
        os.makedirs(os.path.join(self.output_dir, "test"), exist_ok=True)
        model = self.classifier.get_fine_tuned_model().to(device)
        model = model.eval()
        if self.OOD_detector_cfg.type == 'vim':
            self.OOD_detector.initiate_vim(model=model, train_dataloader=self.classifier.in_dist_train_dataloader)

        if self.classifier_cfg.evaluate:
            self.logger.info("Evaluate trained model...")
            self.classifier.evaluate_classifier(best_model=model)
        self.OOD_detector.model = model
        _, _, test_dataloader = create_dataloaders(train_transforms=self.classifier.train_transforms,
                                                   val_transforms=self.classifier.val_transforms,
            data_paths={"train": self.anomaly_detector.output_dir_train_dataset,
                        "val": self.anomaly_detector.output_dir_val_dataset,
                        "test": self.anomaly_detector.output_dir_test_dataset}, dataset_type=DatasetType.NONE,
                        ood_classes_names=self.OOD_detector_cfg.ood_class_names,
                        val_batch_size=self.classifier_cfg.val_batch_size,
            use_weighted_sampler=self.classifier_cfg.weighted_sampler)

        if not os.path.exists(self.OOD_detector.test_dataset_scores_and_labels):
            scores, labels, preds = self.OOD_detector.score_samples(dataloader=test_dataloader,
                                                                               save_outliers=True)
            labels_scores_dict = {'scores': scores.tolist(), 'labels': labels.tolist(), 'preds': preds.tolist()}
            with open(self.OOD_detector.test_dataset_scores_and_labels, 'w') as f:
                json.dump(labels_scores_dict, f, indent=4)
        else:
            with open(self.OOD_detector.test_dataset_scores_and_labels, 'r') as file:
                data = json.load(file)
                scores = torch.tensor(data['scores'])
                labels = torch.tensor(data['labels'])
                preds = torch.tensor(data['preds'])

        # save_TT_1_images(all_scores=scores, all_labels=labels, dataloader=test_dataloader,
        #                  path=self.OOD_detector.test_output, logger=self.logger,
        #                  abnormal_labels=list(test_dataloader.dataset.ood_classes.values()),
        #                  hashmap_locations_and_anomaly_scores_test_file=self.anomaly_detector.hashmap_locations_and_anomaly_scores_test_file,
        #                  visualizer=self.anomaly_detector.bbox_regressor_runner.visualizer,
        #                  test_dataset=self.anomaly_detector.test_dataloader.dataset)

        # show_objects_misclassifed_by_the_dataset(all_scores=scores, dataloader=test_dataloader,
        #                  path=self.OOD_detector.test_output, logger=self.logger,
        #                  hashmap_locations_and_anomaly_scores_test_file=self.anomaly_detector.hashmap_locations_and_anomaly_scores_test_file,
        #                  visualizer=self.anomaly_detector.bbox_regressor_runner.visualizer,
        #                  test_dataset=self.anomaly_detector.test_dataloader.dataset)

        if self.OOD_detector_cfg.save_outliers:
            save_k_outliers(all_scores=scores, all_labels=labels, dataloader=test_dataloader,
                            outliers_path=self.OOD_detector.outliers_path, k=self.OOD_detector_cfg.num_of_outliers,
                            logger=self.logger)

        anomaly_scores, anomaly_scores_conv = retrieve_anomaly_scores_for_test_dataset(test_dataloader,
                                                                  self.anomaly_detector.hashmap_locations_and_anomaly_scores_test_file)

        _, _, eer_threshold = plot_graphs(scores=scores, anomaly_scores=anomaly_scores,
                                         anomaly_scores_conv=anomaly_scores_conv, labels=labels,
                                         path=self.OOD_detector.test_output, title="OOD stage",
                                         abnormal_labels=list(test_dataloader.dataset.ood_classes.values()),
                                         dataset_name="test", ood_mode=True,
                                         labels_to_classes_names=test_dataloader.dataset.labels_to_classes_names,
                                         plot_EER=True, logger=self.logger, OOD_method=self.OOD_detector_cfg.type)

        if self.OOD_detector_cfg.rank_accord_features:
            rank_samples_accord_features(scores, list(test_dataloader.dataset.ood_classes.values()),
                                         eer_threshold, model, test_dataloader, self.OOD_detector.test_output, self.logger)



def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="The relative path to the cfg file")
    args = parser.parse_args()
    pipeline = FullODDPipeline(args)
    pipeline.train()
    pipeline.test()


if __name__ == '__main__':
    main()
