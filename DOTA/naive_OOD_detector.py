import os
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
import numpy as np
import plotly.express as px

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs


def calculate_dataset_scores(test_ds):
    score_dict={}
    for i in range(len(test_ds)):
        in_dist_img = test_ds[i]["pixel_values"].unsqueeze(0)
        sequence_output = model.vit(in_dist_img)[0]
        features = sequence_output[:, 0, :]
        ns = np.Inf
        for j in range(len(train_in_dist_ds)):
            temp_ns = torch.norm(features_dict[j] - features, p=2).item()
            ns = min(ns, temp_ns)
        score_dict[i] = ns
    return score_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help="The relative path to the input database")
    parser.add_argument("-o", "--output_dir", help="The saved model path")
    parser.add_argument('-id', '--in_distribution_list', nargs='+', default=[])
    parser.add_argument('-od', '--out_of_distribution_list', nargs='+', default=[])
    parser.add_argument('-s', '--size', type=int)
    args = parser.parse_args()

    # load in distribution and out of distribution datasets
    in_dist_paths = [os.path.join(args.path, "test", label, "**") for label in args.in_distribution_list]
    out_dist_paths = [os.path.join(args.path, "train", label, "**") for label in args.out_of_distribution_list] + \
                     [os.path.join(args.path, "test", label, "**") for label in args.out_of_distribution_list]
    # loading dataset
    in_bb_dataset = load_dataset("imagefolder", data_files={"in_dist": in_dist_paths})
    out_bb_dataset = load_dataset("imagefolder", data_files={"out_dist": out_dist_paths})
    in_dist_ds = in_bb_dataset['in_dist']
    out_dist_ds = out_bb_dataset['out_dist']

    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)

    in_dist_ds.set_transform(transform=transform)
    out_dist_ds.set_transform(transform=transform)

    in_dist_ds = in_dist_ds.shuffle()
    train_in_dist_ds = in_dist_ds.select(range(int(args.size)))
    test_in_dist_ds = in_dist_ds.select(range(args.size, len(in_dist_ds)))


    labels = in_dist_ds.features['label'].names
    model = ViTForImageClassification.from_pretrained(
        args.output_dir,
        num_labels=len(args.in_distribution_list),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )
    model.eval()
    features_dict = {}
    id_score = {}
    ood_score = {}
    with torch.no_grad():
        for i in range(len(train_in_dist_ds)):
            in_dist_img = in_dist_ds[i]["pixel_values"].unsqueeze(0)
            sequence_output = model.vit(in_dist_img)[0]
            features = sequence_output[:, 0, :]
            features_dict[i] = features

        id_score = calculate_dataset_scores(test_in_dist_ds)
        id_score_fig = px.histogram(id_score.values(), nbins=round(np.sqrt(len(id_score))))
        id_score_fig.write_html(f"./statistics/id_scores.html")
        ood_score = calculate_dataset_scores(out_dist_ds)
        ood_score_fig = px.histogram(ood_score.values(), nbins=round(np.sqrt(len(ood_score))))
        ood_score_fig.write_html(f"./statistics/ood_scores.html")



