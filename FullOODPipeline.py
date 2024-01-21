import json
import logging
import os
from argparse import ArgumentParser
import torch
from mmengine.config import Config
from torch.utils.data import DataLoader

from AnomalyDetector import KNNAnomalyDetector, VitBasedAnomalyDetector
from Classifier import VitClassifier
from OODDetector import ODINOODDetector
from results import plot_graphs
from utils import create_patches_dataloader, calculate_confusion_matrix, create_and_save_confusion_matrix, create_logger

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
        self.classifier.initiate_trainer(classifier_cfg=self.classifier_cfg, dataset_cfg=self.dataset_cfg,
                                         output_dir=self.output_dir)
        if self.classifier_cfg.retrain:
            self.classifier.train()

    def test(self):
        torch.cuda.empty_cache()
        os.makedirs(os.path.join(self.output_dir, "test"), exist_ok=True)
        subtest_dataloader = create_patches_dataloader(type="subtest", dataset_cfg=self.dataset_cfg,
                                                       batch_size=self.classifier_cfg.per_device_eval_batch_size,
                                                       transform=self.classifier.vit_transform,
                                                       output_dir=self.output_dir,
                                                       logger=self.logger,
                                                       is_valid_file_use=True,
                                                       collate_fn=self.classifier.collate_fn)
        model = self.classifier.get_fine_tuned_model().to(device)
        model=model.eval()
        subval_dataloader = DataLoader(dataset=self.classifier.in_distribution_val_dataset,
                                       batch_size=self.classifier_cfg.per_device_eval_batch_size,
                                       shuffle=self.dataset_cfg.shuffle, num_workers=self.dataset_cfg["num_workers"],
                                       collate_fn=self.classifier.collate_fn)
        # get confusion matrix for each label
        calculate_confusion_matrix(dataloader=subval_dataloader, model=model, dataset_name="subval",
                                   path=self.classifier.test_output_dir, logger=self.logger)
        self.OOD_detector.model = model
        if not os.path.exists(subtest_dataloader.dataset.scores_and_labels_file_ood):
            scores, labels, preds = self.OOD_detector.score_samples(dataloader=subtest_dataloader)
            labels_scores_dict = {'scores': scores.tolist(), 'labels': labels.tolist(), 'preds': preds.tolist()}
            with open(subtest_dataloader.dataset.scores_and_labels_file_ood, 'w') as f:
                json.dump(labels_scores_dict, f, indent=4)
        else:
            with open(subtest_dataloader.dataset.scores_and_labels_file_ood, 'r') as file:
                data = json.load(file)
                scores = torch.tensor(data['scores'])
                labels = torch.tensor(data['labels'])
                preds = torch.tensor(data['preds'])
        create_and_save_confusion_matrix(all_labels=labels, all_preds=preds, path=self.classifier.test_output_dir,
                                         dataset_name="subtest", logger=self.logger)
        plot_graphs(scores=scores, labels=labels, path=os.path.join(self.output_dir, "test/OOD"), title="OOD stage",
                    abnormal_labels=list(subtest_dataloader.dataset.ood_classes.values()), dataset_name="test",
                    ood_mode=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="The relative path to the cfg file")
    args = parser.parse_args()
    pipeline = FullODDPipeline(args)
    pipeline.train()
    pipeline.test()


if __name__ == '__main__':
    main()
