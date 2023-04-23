import os
from argparse import ArgumentParser
from collections import Counter
from typing import Dict

import cv2
import plotly.express as px

from DOTA_devkit.dota_utils import dots4ToRec4
from dataset import DotaDataset

SIZE = 224


class PatchesDatasetCreator:
    dota_dataset: DotaDataset
    patch_id_to_label: Dict[int, str]
    path_to_save_images: str

    def __init__(self, dota_dataset, args):
        self.dota_dataset: DotaDataset = dota_dataset
        self.patch_id_to_label = {}
        self.threshold = int(SIZE * args.threshold)
        folder_path = os.path.join(args.path, "bb_datasets_thresh_" + str(self.threshold))
        self.path = os.path.join(folder_path, "train") if dota_dataset.is_train else os.path.join(folder_path, "test")
        self.histogram_file_path = f"./statistics/classes_histogram_train_thresh_{self.threshold}.html" \
            if dota_dataset.is_train else f"./statistics/classes_histogram_test_thresh_{self.threshold}.html"
        os.makedirs(self.path, exist_ok=True)
        os.makedirs("./statistics", exist_ok=True)
        for cls in self.dota_dataset.catToImgs.keys():
            os.makedirs(os.path.join(self.path, cls), exist_ok=True)

    def __call__(self):
        ind = 0
        for i, (image, anns) in enumerate(self.dota_dataset):
            for ann in anns:
                xmin, ymin, xmax, ymax = dots4ToRec4(ann['poly'])
                if self.threshold < min(xmax - xmin, ymax - ymin):
                    square_length_y = abs(xmax - xmin) - abs(ymax - ymin) if abs(xmax - xmin) > abs(ymax - ymin) else 0
                    square_length_x = abs(ymax - ymin) - abs(xmax - xmin) if abs(xmax - xmin) < abs(ymax - ymin) else 0
                    self.patch_id_to_label[ind] = ann['name']
                    if image is None:
                        continue
                    cropped_img = image[int(ymin - square_length_y / 2):int(ymax + square_length_y / 2),
                                  int(xmin - square_length_x / 2):int(xmax + square_length_x / 2)]
                    if cropped_img.size != 0:
                        cv2.imwrite(os.path.join(self.path, ann['name'], f"{ind}.jpg"), cropped_img)
                        ind += 1

        value_counts = Counter(self.patch_id_to_label.values())
        print(value_counts)
        fig = px.histogram(list(dict(value_counts).items()), x=0, y=1, labels={'0': 'labels', '1': 'labels'})
        fig.show()
        fig.write_html(self.histogram_file_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help="The relative path to save the output database")
    parser.add_argument("-t", "--train", action='store_true', help="True if creating train dataset "
                                                                   "otherwise creating test dataset")
    parser.add_argument("-th", "--threshold",
                        type=float, default=0.25,
                        help="The default ratio of the minimal blocking square's side of the BB. "
                             "this ratio is taken from the SIZE parameter.")
    args = parser.parse_args()
    dota_dataset = DotaDataset(is_train=args.train)
    creator = PatchesDatasetCreator(dota_dataset=dota_dataset, args=args)
    creator()
