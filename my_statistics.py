from pathlib import Path
import torch
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


def get_unfound_fg_areas_n_aspect_ratio(fg_gt_bboxes: torch.tensor, founded_fg_indices: torch.tensor):
    unfounded_indices = torch.arange(0, fg_gt_bboxes.shape[0], 1)
    unfounded_indices = unfounded_indices[~torch.eq(unfounded_indices[:, None], founded_fg_indices).any(dim=1)]
    unfounded_areas = fg_gt_bboxes[unfounded_indices].areas
    unfounded_heights = fg_gt_bboxes[unfounded_indices].heights
    unfounded_widths = fg_gt_bboxes[unfounded_indices].widths
    unfounded_aspect_ratio = unfounded_widths / unfounded_heights
    founded_areas = fg_gt_bboxes[founded_fg_indices].areas
    founded_heights = fg_gt_bboxes[founded_fg_indices].heights
    founded_widths = fg_gt_bboxes[founded_fg_indices].widths
    founded_aspect_ratio = founded_widths / founded_heights
    return unfounded_areas, unfounded_aspect_ratio, founded_areas, founded_aspect_ratio


def plot_fg_statistics(all_fg_areas: torch.tensor, all_fg_aspect_ratio: torch.tensor,
                       is_unfound: bool):
    elements = all_fg_areas.size(0)
    areas_title = f"Unfound area sizes histogram - {elements}" if is_unfound else f"Founded area sizes histogram - {elements}"
    areas_fig = px.histogram(np.array(all_fg_areas),
                             labels={'value': 'area size (log scale)', 'count': 'count'},
                             nbins=4*round(np.sqrt(len(all_fg_areas))),
                             title=areas_title)
    areas_fig.show()
    aspect_ratio_title = "Unfound aspect ratio histogram" if is_unfound else "Founded aspect ratio histogram"
    aspect_ratio_fig = px.histogram(np.array(all_fg_aspect_ratio),
                                    labels={'value': 'aspect_ratio (log scale)', 'count': 'count'},
                                    nbins=4 * round(np.sqrt(len(all_fg_aspect_ratio))),
                                    title=aspect_ratio_title)
    aspect_ratio_fig.show()


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
