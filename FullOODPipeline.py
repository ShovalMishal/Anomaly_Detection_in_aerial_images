import json

import os
from argparse import ArgumentParser
import torch
from mmengine.config import Config

from AnomalyDetector import VitBasedAnomalyDetector
from Classifier import VitClassifier
from OODDetector import ODINOODDetector
from Classifier import create_dataloaders, DatasetType
from results import plot_graphs
from utils import create_logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FullODDPipeline:
    def __init__(self, args):
        self.cfg = Config.fromfile(args.config)
        self.output_dir = self.cfg.output_dir
        os.makedirs(os.path.join(self.output_dir), exist_ok=True)
        self.logger = create_logger(os.path.join(self.output_dir, "full_ood_pipeline_log.log"))
        anomaly_detector_cfg = self.cfg.get("anomaly_detector_cfg")
        self.classifier_cfg = self.cfg.get("classifier_cfg")
        self.OOD_detector_cfg = self.cfg.get("OOD_detector_cfg")
        self.anomaly_detector = {'vit_based_anomaly_detector': VitBasedAnomalyDetector}[anomaly_detector_cfg.type]
        self.anomaly_detector = self.anomaly_detector(anomaly_detector_cfg,
                                                      self.output_dir, self.logger)
        self.classifier = {'vit': VitClassifier}[self.classifier_cfg.type]
        self.classifier = self.classifier(self.output_dir, self.classifier_cfg, self.logger)
        self.OOD_detector = {'ODIN': ODINOODDetector}[self.OOD_detector_cfg.type]
        self.OOD_detector = self.OOD_detector(output_dir=self.output_dir)

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
        self.logger.info("Evaluate trained model...")
        self.classifier.evaluate_classifier(best_model=model)
        self.OOD_detector.model = model
        _, _, test_dataloader = create_dataloaders(
            data_paths={"train": self.anomaly_detector.output_dir_train_dataset,
                        "val": self.anomaly_detector.output_dir_val_dataset,
                        "test": self.anomaly_detector.output_dir_test_dataset}, dataset_type=DatasetType.NONE,
                        ood_classes_names=self.OOD_detector_cfg.ood_class_names)

        if not os.path.exists(self.OOD_detector.test_dataset_scores_and_labels):
            scores, labels, preds = self.OOD_detector.score_samples(dataloader=test_dataloader)
            labels_scores_dict = {'scores': scores.tolist(), 'labels': labels.tolist(), 'preds': preds.tolist()}
            with open(self.OOD_detector.test_dataset_scores_and_labels, 'w') as f:
                json.dump(labels_scores_dict, f, indent=4)
        else:
            with open(self.OOD_detector.test_dataset_scores_and_labels, 'r') as file:
                data = json.load(file)
                scores = torch.tensor(data['scores'])
                labels = torch.tensor(data['labels'])
                preds = torch.tensor(data['preds'])

        plot_graphs(scores=scores, labels=labels, path= self.OOD_detector.test_output, title="OOD stage",
                    abnormal_labels=list(test_dataloader.dataset.ood_classes.values()), dataset_name="test",
                    ood_mode=True, labels_to_classes_names=test_dataloader.dataset.labels_to_classe_names)



def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="The relative path to the cfg file")
    args = parser.parse_args()
    pipeline = FullODDPipeline(args)
    pipeline.train()
    pipeline.test()


if __name__ == '__main__':
    main()
