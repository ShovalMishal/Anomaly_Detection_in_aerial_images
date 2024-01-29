import os
from enum import Enum
from typing import Union, Optional, Dict, Callable, List, Tuple

import torch
from plotly.figure_factory import np
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from OOD_Upper_Bound.finetune_vit_classifier import train_classifier, ViTLightningModule
from OOD_Upper_Bound.ood_and_id_dataset import OODAndIDDataset
from OOD_Upper_Bound.split_dataset_to_id_and_ood import create_ood_id_dataset
from results import plot_confusion_matrix
from utils import create_patches_dataset, eval_model
from datasets import load_metric, Dataset
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer, PreTrainedModel, \
    DataCollator, PreTrainedTokenizerBase, EvalPrediction, TrainerCallback
from CustomSampler import CustomSampler


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


class CustomTrainer(Trainer):
    def __init__(
            self,
            sampler_type: str,
            sampler_cfg: Dict,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,

    ):
        super().__init__(model=model, args=args,
                         data_collator=data_collator,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         tokenizer=tokenizer,
                         model_init=model_init,
                         compute_metrics=compute_metrics,
                         callbacks=callbacks,
                         optimizers=optimizers,
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics)
        self.sampler = {"CustomSampler": CustomSampler, "WeightedRandomSampler": WeightedRandomSampler}[sampler_type]
        self.sampler_type = sampler_type
        self.sampler_cfg = sampler_cfg

    def _get_train_sampler(self) -> torch.utils.data.Sampler:
        # Create a stratified sampler
        labels = [sample["labels"] for sample in self.train_dataset]
        sampler = None
        if self.sampler_type == "CustomSampler":
            sampler = self.sampler(labels=labels, num_samples=len(self.train_dataset), batch_size=16,
                                   labels_frequency_inbatch=self.sampler_cfg.custom_sampler_labels_frequency,
                                   shuffle=True)
        elif self.sampler_type == "WeightedRandomSampler":
            sampler = self.sampler(weights=list(self.train_dataset.weights.values()), num_samples=len(self.train_dataset))
        # Return the stratified sampler
        return sampler



def compute_metrics(p):
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

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
    train_ds = OODAndIDDataset(root_dir=os.path.join(data_paths["train"], "images") if dataset_type is DatasetType.NONE
    else os.path.join(data_paths["train"], f"{dataset_type.value}_dataset"),
                               dataset_type="train",
                               transform=_train_transforms,
                               ood_classes_names=ood_classes_names)
    val_ds = OODAndIDDataset(
        root_dir=os.path.join(data_paths["val"], "images") if dataset_type is DatasetType.NONE else
        os.path.join(data_paths["val"], f"{dataset_type.value}_dataset"),
        dataset_type="val",
        transform=_val_transforms,
        ood_classes_names=ood_classes_names)
    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size,
                                  num_workers=12)
    val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=12)
    return train_dataloader, val_dataloader

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
        self.in_dist_train_dataloader, self.in_dist_val_dataloader = create_dataloaders(
            data_paths={"train": os.path.join(data_source, "train_ood_id_split"),
                        "val": os.path.join(data_source, "val_ood_id_split")}, dataset_type=DatasetType.IN_DISTRIBUTION)

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
                                   test_dataloader=self.in_dist_val_dataloader,
                                   id2label=self.id2label,
                                   label2id=self.label2id, num_labels=len(self.in_dist_val_dataloader), )



    def train(self):
        self.logger.info(f"Starting to train the ViT classifier\n")
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=3,
            strict=False,
            verbose=False,
            mode='min'
        )
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
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
            test_dataloader=self.in_dist_val_dataloader,
            id2label=self.id2label,
            label2id=self.label2id,
            num_labels=len(self.in_dist_val_dataloader)
        )
        return model

    def evaluate_classifier(self, best_model):
        all_preds, all_labels = eval_model(dataloader=self.classifier.in_dist_val_dataloader, model=best_model)
        res_confusion_matrix = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(confusion_matrix=res_confusion_matrix, classes=self.in_dist_train_dataloader.dataset.classes,
                              normalize=True,
                              output_dir=self.test_output_dir)
        plot_confusion_matrix(confusion_matrix=res_confusion_matrix, classes=self.in_dist_train_dataloader.dataset.classes,
                              normalize=False,
                              output_dir=self.test_output_dir)
