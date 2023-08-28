from argparse import ArgumentParser
import torch
from mmengine.config import Config
from Anomaly_Detector import KNNAnomalyDetetctor, AnomalyDetector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FullODDPipeline:
    def __init__(self, args):
        self.output_dir = args.output_dir
        self.cfg = Config.fromfile(args.config)
        self.dataset_cfg = self.cfg.get("patches_dataset")
        anomaly_detector_cfg = self.cfg.get("anomaly_detector_cfg")
        embedder_cfg = self.cfg.get("embedder_cfg")
        dataset_cfg = self.cfg.get("patches_dataset")
        self.anomaly_detector = KNNAnomalyDetetctor(embedder_cfg, anomaly_detector_cfg, self.output_dir, dataset_cfg) \
            if anomaly_detector_cfg.type == "knn" else AnomalyDetector()

    def train(self):
        self.anomaly_detector.train()

    def test(self):
        pass


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="The relative path to the cfg file")
    parser.add_argument("-o", "--output_dir", help="Statistics output dir")
    args = parser.parse_args()
    pipeline = FullODDPipeline(args)
    pipeline.train()


if __name__ == '__main__':
    main()
