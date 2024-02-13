import argparse
import json
import os
import sys
from sklearn.metrics import confusion_matrix
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python module search path
sys.path.insert(0, parent_dir)
from OODDetector import ODINOODDetector
from OOD_Upper_Bound.finetune_vit_classifier import create_dataloaders, train_classifier, DatasetType
from OOD_Upper_Bound.split_dataset_to_id_and_ood import create_ood_id_dataset
from results import plot_confusion_matrix, plot_graphs
from utils import eval_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Running OOD Detector - stage 2')
    parser.add_argument('--data_source', help='Source root directory')
    parser.add_argument('--output', help='Source root directory', default="./")
    parser.add_argument('--ood_class_names', nargs='+', help='List of out of distribution class names')
    parser.add_argument('--train', action='store_true', help='Train the model if true, otherwise load it.')
    args = parser.parse_args()
    data_source = args.data_source
    ood_class_names = args.ood_class_names
    train = args.train
    output = args.output
    return data_source, ood_class_names, train, output


def ood_detector_stage(data_source, train, ood_class_names, output):
    # Create dataloaders
    in_dist_train_dataloader, in_dist_val_dataloader = create_dataloaders(
        data_paths={"train": os.path.join(data_source, "train_ood_id_split"),
                    "val": os.path.join(data_source, "val_ood_id_split")}, dataset_type=DatasetType.IN_DISTRIBUTION)
    # Train classifier
    model = train_classifier(in_dist_train_dataloader, in_dist_val_dataloader, output, num_labels=len(in_dist_train_dataloader.dataset.classes), train=train)

    # Create output folders
    os.makedirs(os.path.join(output, "upper_bound_output/OOD_Detector"), exist_ok=True)
    os.makedirs(os.path.join(output, "upper_bound_output/Classifier"), exist_ok=True)
    # Create OOD detector
    OOD_detector = ODINOODDetector(output_dir=os.path.join(output, "upper_bound_output/OOD_Detector"))
    OOD_detector.model = model
    # Eval trained model
    all_preds, all_labels = eval_model(dataloader=in_dist_val_dataloader, model=model)
    res_confusion_matrix = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(confusion_matrix=res_confusion_matrix, classes=in_dist_train_dataloader.dataset.classes,
                          normalize=True,
                          output_dir=os.path.join(output, "upper_bound_output/Classifier"))
    plot_confusion_matrix(confusion_matrix=res_confusion_matrix, classes=in_dist_train_dataloader.dataset.classes,
                          normalize=False,
                          output_dir=os.path.join(output, "upper_bound_output/Classifier"))
    # Calculate OOD detector scores
    train_dataloader, val_dataloader = create_dataloaders(
        data_paths={"train": os.path.join(data_source, "train"),
                    "val": os.path.join(data_source, "val")}, dataset_type=DatasetType.NONE, ood_classes_names=ood_class_names)
    scores, labels, preds = OOD_detector.score_samples(dataloader=val_dataloader)
    labels_scores_dict = {'scores': scores.tolist(), 'labels': labels.tolist(), 'preds': preds.tolist()}
    with open(os.path.join(output, "upper_bound_output/OOD_Detector/val_dataset_scores_and_labels.json", 'w')) as f:
        json.dump(labels_scores_dict, f, indent=4)
    # Plot OOD detector results
    plot_graphs(scores=scores, labels=labels, path=os.path.join(output, "upper_bound_output/OOD_Detector"),
                title="OOD stage", abnormal_labels=list(val_dataloader.dataset.ood_classes.values()), dataset_name="val",
                ood_mode=True, labels_to_classes_names=val_dataloader.dataset.labels_to_classe_names)


if __name__ == '__main__':
    data_source, ood_class_names, train, output = parse_args()
    create_ood_id_dataset(src_root=os.path.join(data_source, "train"),
                          target_root=os.path.join(data_source, "train_ood_id_split"),
                          ood_classes=ood_class_names)
    create_ood_id_dataset(src_root=os.path.join(data_source, "val"),
                          target_root=os.path.join(data_source, "val_ood_id_split"),
                          ood_classes=ood_class_names)
    ood_detector_stage(data_source, train, ood_class_names, output)
