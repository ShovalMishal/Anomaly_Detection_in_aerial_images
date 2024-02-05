import os
from enum import Enum
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from OOD_Upper_Bound.finetune_vit_classifier import ViTLightningModule
from OOD_Upper_Bound.ood_and_id_dataset import OODAndIDDataset
from OOD_Upper_Bound.split_dataset_to_id_and_ood import create_ood_id_dataset
from results import plot_confusion_matrix
from utils import eval_model
from transformers import ViTImageProcessor



class DatasetType(Enum):
    IN_DISTRIBUTION = "id"
    OUT_OF_DISTRIBUTION = "ood"
    NONE = ""


class Classifier():
    def __init__(self, output_dir, classifier_cfg, logger):
        self.classifier_cfg = classifier_cfg
        self.train_output_dir = os.path.join(output_dir, classifier_cfg.train_output_dir)
        self.test_output_dir = os.path.join(output_dir, classifier_cfg.test_output_dir)
        os.makedirs(self.test_output_dir, exist_ok=True)
        self.logger = logger

    def train(self):
        pass

    def test(self):
        pass


def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def create_dataloaders(data_paths, dataset_type: DatasetType, ood_classes_names=[]):
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = processor.size["height"]
    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )
    train_batch_size = 100
    eval_batch_size = 100
    train_ds = OODAndIDDataset(root_dir=os.path.join(data_paths["train"]) if dataset_type is DatasetType.NONE
    else os.path.join(data_paths["train"], f"{dataset_type.value}_dataset"),
                               dataset_type="train",
                               transform=_train_transforms,
                               ood_classes_names=ood_classes_names)
    val_ds = OODAndIDDataset(
        root_dir=os.path.join(data_paths["val"]) if dataset_type is DatasetType.NONE else
        os.path.join(data_paths["val"], f"{dataset_type.value}_dataset"),
        dataset_type="val",
        transform=_val_transforms,
        ood_classes_names=ood_classes_names)
    test_ds = OODAndIDDataset(
        root_dir=os.path.join(data_paths["test"]) if dataset_type is DatasetType.NONE else
        os.path.join(data_paths["test"], f"{dataset_type.value}_dataset"),
        dataset_type="test",
        transform=_val_transforms,
        ood_classes_names=ood_classes_names)
    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size,
                                  num_workers=12)
    val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=12)
    test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=12)
    return train_dataloader, val_dataloader, test_dataloader

class VitClassifier(Classifier):
    def __init__(self, output_dir, classifier_cfg, logger):
        super().__init__(output_dir, classifier_cfg, logger)



    def initiate_trainer(self, data_source, ood_class_names):
        self.logger.info(f"Initializing classifier\n")
        # create OOD and ID datasets folders according to Anomaly Detection stage results
        create_ood_id_dataset(src_root=os.path.join(data_source, "train"),
                              target_root=os.path.join(data_source, "train_ood_id_split"),
                              ood_classes=ood_class_names)
        create_ood_id_dataset(src_root=os.path.join(data_source, "val"),
                              target_root=os.path.join(data_source, "val_ood_id_split"),
                              ood_classes=ood_class_names)
        create_ood_id_dataset(src_root=os.path.join(data_source, "test"),
                              target_root=os.path.join(data_source, "test_ood_id_split"),
                              ood_classes=ood_class_names)
        self.in_dist_train_dataloader, self.in_dist_val_dataloader, self.in_dist_test_dataloader = create_dataloaders(
            data_paths={"train": os.path.join(data_source, "train_ood_id_split"),
                        "val": os.path.join(data_source, "val_ood_id_split"),
                        "test": os.path.join(data_source, "test_ood_id_split"),
                        }, dataset_type=DatasetType.IN_DISTRIBUTION)

        self.id2label = self.in_dist_train_dataloader.dataset.labels_to_classe_names
        self.label2id = self.in_dist_train_dataloader.dataset.classes_names_to_labels

        train_batch_size = 100

        batch = next(iter(self.in_dist_train_dataloader))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)

        assert batch['pixel_values'].shape == (train_batch_size, 3, 224, 224)
        assert batch['labels'].shape == (train_batch_size,)


        self.model = ViTLightningModule(train_dataloader=self.in_dist_train_dataloader, val_dataloader=self.in_dist_val_dataloader,
                                   test_dataloader=self.in_dist_test_dataloader,
                                   id2label=self.id2label,
                                   label2id=self.label2id, num_labels=len(self.id2label), )



    def train(self):
        self.logger.info(f"Starting to train the ViT classifier\n")
        early_stop_callback = EarlyStopping(
            monitor='validation_loss',
            patience=3,
            strict=False,
            verbose=False,
            mode='min'
        )
        checkpoint_callback = ModelCheckpoint(
            monitor='validation_loss',
            mode='min',
            save_top_k=1,
            dirpath=self.classifier_cfg.checkpoint_path,
            filename='best_model',
        )
        trainer = Trainer(num_nodes=1, callbacks=[early_stop_callback, checkpoint_callback])
        trainer.fit(self.model)

        """Finally, let's test the trained model on the test set:"""

        trainer.test()

    def get_fine_tuned_model(self):
        model = ViTLightningModule.load_from_checkpoint(
            os.path.join(self.classifier_cfg.checkpoint_path, f"best_model.ckpt"),
            train_dataloader=self.in_dist_train_dataloader,
            val_dataloader=self.in_dist_val_dataloader,
            test_dataloader=self.in_dist_test_dataloader,
            id2label=self.id2label,
            label2id=self.label2id,
            num_labels=len(self.label2id)
        )
        return model

    def evaluate_classifier(self, best_model):
        all_preds, all_labels = eval_model(dataloader=self.in_dist_test_dataloader, model=best_model)
        res_confusion_matrix = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(confusion_matrix=res_confusion_matrix, classes=self.in_dist_train_dataloader.dataset.classes,
                              normalize=True,
                              output_dir=self.test_output_dir)
        plot_confusion_matrix(confusion_matrix=res_confusion_matrix, classes=self.in_dist_train_dataloader.dataset.classes,
                              normalize=False,
                              output_dir=self.test_output_dir)
