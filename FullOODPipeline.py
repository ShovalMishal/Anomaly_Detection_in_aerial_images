import os
from argparse import ArgumentParser
import torch
from mmengine.config import Config
from AnomalyDetector import KNNAnomalyDetector, AnomalyDetector
from Classfier import VitClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FullODDPipeline:
    def __init__(self, args):
        self.output_dir = args.output_dir
        self.cfg = Config.fromfile(args.config)
        self.dataset_cfg = self.cfg.get("patches_dataset")
        anomaly_detector_cfg = self.cfg.get("anomaly_detector_cfg")
        embedder_cfg = self.cfg.get("embedder_cfg")
        self.dataset_cfg = self.cfg.get("patches_dataset")
        self.classifier_cfg = self.cfg.get("classifier_cfg")
        self.anomaly_detector = {'knn': KNNAnomalyDetector}[anomaly_detector_cfg.type]
        self.anomaly_detector = self.anomaly_detector(embedder_cfg, anomaly_detector_cfg, self.dataset_cfg, self.output_dir)
        self.classifier = {'vit': VitClassifier}[self.classifier_cfg.type]
        self.classifier = self.classifier()

    def train(self):
        os.makedirs(os.path.join(self.output_dir, "train"), exist_ok=True)
        self.anomaly_detector.train()
        self.classifier.initiate_trainer(classifier_cfg=self.classifier_cfg, dataset_cfg=self.dataset_cfg, output_dir=self.output_dir)
        self.classifier.train()

    def test(self):
        os.makedirs(os.path.join(self.output_dir, "test"), exist_ok=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="The relative path to the cfg file")
    parser.add_argument("-o", "--output_dir", help="Statistics output dir")
    args = parser.parse_args()
    pipeline = FullODDPipeline(args)
    pipeline.train()


if __name__ == '__main__':
    main()
