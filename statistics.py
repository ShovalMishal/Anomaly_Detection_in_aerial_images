from pathlib import Path

import plotly.express as px
from dota_dataset import DotaDataset
import numpy as np


def create_size_histogram_accord_class(dota_dataset: DotaDataset, classes_names=[]):
    img_ids = dota_dataset.get_img_ids_accord_categories(catNms=classes_names)
    sizes = []
    for img_id in img_ids:
        anns = dota_dataset.loadAnns(catNms=classes_names, imgId=img_id)
        for ann in anns:
            sizes.append(ann['area'])
    fig_value = classes_names[0] if len(classes_names) == 1 else "all_classes"
    fig = px.histogram(np.array(sizes), labels={'value': f'{fig_value}', 'count': 'area size (log scale)'},
                       nbins=round(np.sqrt(len(sizes))),
                       title="area sizes histogram")
    fig.show()
    fig.write_html(f"./statistics/area_sizes_{fig_value}.html")


def create_count_histogram_accord_classes(dota_dataset: DotaDataset, classes_names=[]):
    counter = dict(zip(classes_names, [0] * len(classes_names)))
    img_ids = dota_dataset.get_img_ids_accord_categories(catNms=[])
    for img_id in img_ids:
        anns = dota_dataset.loadAnns(catNms=classes_names, imgId=img_id)
        for ann in anns:
            counter[ann['name']] += 1
    fig = px.histogram(list(counter.items()), x=0, y=1, labels={'0': 'labels', '1': 'labels'})
    fig.show()
    fig.write_html(f"./statistics/classes_counts_histogram.html")


def run_statistics():
    Path("./statistics").mkdir(parents=True, exist_ok=True)
    dota_dataset = DotaDataset()
    classes = list(dota_dataset.catToImgs.keys())
    for cls in classes:
        create_size_histogram_accord_class(dota_dataset=dota_dataset, classes_names=[cls])
    # histogram for al classes
    create_size_histogram_accord_class(dota_dataset=dota_dataset, classes_names=[])
    create_count_histogram_accord_classes(dota_dataset=dota_dataset, classes_names=classes)
    print(classes)


if __name__ == '__main__':
    run_statistics()
