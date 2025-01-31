import os
from enum import Enum
from types import SimpleNamespace

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor, RandomVerticalFlip)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from Classifiers.finetune_resnet50_classifier import ResNet50LightningModule
from Classifiers.finetune_vit_classifier import ViTLightningModule
from ood_and_id_dataset import OODAndIDDataset
from utils import create_ood_id_dataset
from plot_results import plot_confusion_matrix
from resnet_pytorch_small_images.train import create_and_train_model
from resnet_pytorch_small_images.utils import get_network, best_acc_weights, most_recent_folder
from utils import eval_model, ResizeLargestAndPad
from transformers import ViTImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
class DatasetType(Enum):
    IN_DISTRIBUTION = "id"
    OUT_OF_DISTRIBUTION = "ood"
    NONE = ""


class Classifier:
    def __init__(self, output_dir, classifier_cfg, logger, current_run_name, data_source, ood_class_names):
        self.in_dist_train_dataloader = None
        self.in_dist_test_dataloader = None
        self.current_run_name = current_run_name
        self.classifier_cfg = classifier_cfg
        self.data_source = data_source
        self.ood_class_names = ood_class_names
        self.train_output_dir = os.path.join(output_dir, classifier_cfg.train_output_dir, classifier_cfg.type, current_run_name)
        self.checkpoint_path = os.path.join(self.train_output_dir, self.classifier_cfg.checkpoint_path)
        self.test_output_dir = os.path.join(output_dir, classifier_cfg.test_output_dir, classifier_cfg.type, current_run_name)
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.train_output_dir, exist_ok=True)
        self.logger = logger

    def preperae_id_ood_datasets(self, data_source, ood_class_names):
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


    def evaluate_classifier(self, best_model):
        self.logger.info("Evaluate trained model...")
        if self.classifier_cfg.evaluate:
            all_preds, all_labels = eval_model(dataloader=self.in_dist_test_dataloader, model=best_model, cache_dir=self.test_output_dir)
            res_confusion_matrix = confusion_matrix(all_labels, all_preds)
            plot_confusion_matrix(confusion_matrix=res_confusion_matrix,
                                  classes=self.in_dist_train_dataloader.dataset.classes,
                                  normalize=True,
                                  output_dir=self.test_output_dir)
            plot_confusion_matrix(confusion_matrix=res_confusion_matrix,
                                  classes=self.in_dist_train_dataloader.dataset.classes,
                                  normalize=False,
                                  output_dir=self.test_output_dir)

    def initiate_trainer(self):
        pass

    def train(self):
        pass


def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def create_dataloaders(train_transforms, val_transforms, data_paths, dataset_type: DatasetType,
                       train_batch_size=100, val_batch_size=100,
                       dataloader_num_workers=4, ood_classes_names=None, use_weighted_sampler=False):
    if ood_classes_names is None:
        ood_classes_names = []
    train_ds = OODAndIDDataset(root_dir=os.path.join(data_paths["train"]) if dataset_type is DatasetType.NONE
    else os.path.join(data_paths["train"], f"{dataset_type.value}_dataset"),
                               dataset_type="train",
                               transform=train_transforms,
                               ood_classes_names=ood_classes_names)
    val_ds = OODAndIDDataset(
        root_dir=os.path.join(data_paths["val"]) if dataset_type is DatasetType.NONE else
        os.path.join(data_paths["val"], f"{dataset_type.value}_dataset"),
        dataset_type="val",
        transform=val_transforms,
        ood_classes_names=ood_classes_names)
    test_ds = OODAndIDDataset(
        root_dir=os.path.join(data_paths["test"]) if dataset_type is DatasetType.NONE else
        os.path.join(data_paths["test"], f"{dataset_type.value}_dataset"),
        dataset_type="test",
        transform=val_transforms,
        ood_classes_names=ood_classes_names)

    train_sampler = WeightedRandomSampler(train_ds.weights, len(train_ds), replacement=True) if use_weighted_sampler else None
    train_dataloader = DataLoader(train_ds,
                                  batch_size=train_batch_size,
                                  num_workers=dataloader_num_workers, sampler=train_sampler, shuffle=True)
    val_sampler = WeightedRandomSampler(val_ds.weights, len(val_ds), replacement=True) if use_weighted_sampler else None
    val_dataloader = DataLoader(val_ds,
                                batch_size=val_batch_size,
                                num_workers=dataloader_num_workers, sampler=val_sampler)

    # test_sampler = WeightedRandomSampler(test_ds.weights, len(test_ds), replacement=True) if use_weighted_sampler else None
    test_dataloader = DataLoader(test_ds,
                                 batch_size=val_batch_size,
                                 num_workers=dataloader_num_workers)

    return train_dataloader, val_dataloader, test_dataloader


class ResNetClassifier(Classifier):
    def __init__(self, output_dir, classifier_cfg, logger, current_run_name, data_source, ood_class_names):
        super().__init__(output_dir, classifier_cfg, logger, current_run_name, data_source, ood_class_names)
        self.resize = None

    def get_resnet_transforms(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.train_transforms = Compose([
            ResizeLargestAndPad(self.resize),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
            Normalize(mean, std)
        ])

        self.val_transforms = Compose([
            ResizeLargestAndPad(self.resize),
            ToTensor(),
            Normalize(mean, std)
        ])


class ResNet50Classifier(ResNetClassifier):
    def __init__(self, output_dir, classifier_cfg, logger, current_run_name):
        super().__init__(output_dir, classifier_cfg, logger, current_run_name)
        self.resize=224

    def train(self):
        self.initiate_trainer()
        if self.classifier_cfg.retrain:
            self.logger.info(f"Starting to train the ResNet50 classifier\n")
            early_stop_callback = EarlyStopping(
                monitor='validation_loss',
                patience=3,
                strict=True,
                verbose=True,
                mode='min'
            )
            checkpoint_callback = ModelCheckpoint(
                monitor='validation_loss',
                mode='min',
                save_top_k=1,
                dirpath=self.checkpoint_path,
                filename='best_model',
            )

            logger = TensorBoardLogger(save_dir=os.path.join(self.train_output_dir, "tb_logs"), name=self.current_run_name)
            trainer = Trainer(num_nodes=1, max_epochs=self.classifier_cfg.max_epoch,
                              callbacks=[early_stop_callback, checkpoint_callback], logger=logger)
            ckpt_path = os.path.join(self.checkpoint_path, "best_model.ckpt") if self.classifier_cfg.resume else None
            trainer.fit(self.model, ckpt_path=ckpt_path)

            """Finally, let's test the trained model on the test set:"""

            results = trainer.test()
            print(results)

    def initiate_trainer(self):
        self.logger.info(f"Initializing classifier\n")
        self.preperae_id_ood_datasets(self.data_source, self.ood_class_names)
        self.get_resnet_transforms()
        self.in_dist_train_dataloader, self.in_dist_val_dataloader, self.in_dist_test_dataloader = create_dataloaders(
            train_transforms=self.train_transforms, val_transforms=self.val_transforms,
            data_paths={"train": os.path.join(self.data_source, "train_ood_id_split"),
                        "val": os.path.join(self.data_source, "val_ood_id_split"),
                        "test": os.path.join(self.data_source, "test_ood_id_split"),
                        }, dataset_type=DatasetType.IN_DISTRIBUTION, val_batch_size=self.classifier_cfg.val_batch_size,
            train_batch_size=self.classifier_cfg.train_batch_size,
            dataloader_num_workers=self.classifier_cfg.dataloader_num_workers,
            use_weighted_sampler=self.classifier_cfg.weighted_sampler)

        batch = next(iter(self.in_dist_train_dataloader))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
        assert batch['pixel_values'].shape == (self.classifier_cfg.train_batch_size, 3, self.resize, self.resize)
        assert batch['labels'].shape == (self.classifier_cfg.train_batch_size,)
        self.id2label = self.in_dist_train_dataloader.dataset.labels_to_classes_names
        self.label2id = self.in_dist_train_dataloader.dataset.classes_names_to_labels

        self.model = ResNet50LightningModule(train_dataloader=self.in_dist_train_dataloader,
                                        val_dataloader=self.in_dist_val_dataloader,
                                        test_dataloader=self.in_dist_test_dataloader,
                                        loss_class_weights=self.classifier_cfg.loss_class_weights,
                                        max_epochs=self.classifier_cfg.max_epoch,
                                        num_labels=len(self.id2label))

        self.model = self.model.to(device)



    def get_fine_tuned_model(self):
        model = ResNet50LightningModule.load_from_checkpoint(
            os.path.join(self.checkpoint_path, f"best_model.ckpt"),
            train_dataloader=self.in_dist_train_dataloader,
            val_dataloader=self.in_dist_val_dataloader,
            test_dataloader=self.in_dist_test_dataloader,
            id2label=self.id2label,
            label2id=self.label2id,
            num_labels=len(self.label2id)
        )
        return model


class ResNet18Classifier(ResNetClassifier):
    def __init__(self, output_dir, classifier_cfg, logger, current_run_name, data_source, ood_class_names):
        super().__init__(output_dir, classifier_cfg, logger, current_run_name, data_source, ood_class_names)
        self.resize=32

    def initiate_trainer(self):
        self.logger.info(f"Initializing classifier\n")
        self.preperae_id_ood_datasets(self.data_source, self.ood_class_names)
        self.get_resnet_transforms()
        self.in_dist_train_dataloader, self.in_dist_val_dataloader, self.in_dist_test_dataloader = create_dataloaders(
            train_transforms=self.train_transforms, val_transforms=self.val_transforms,
            data_paths={"train": os.path.join(self.data_source, "train_ood_id_split"),
                        "val": os.path.join(self.data_source, "val_ood_id_split"),
                        "test": os.path.join(self.data_source, "test_ood_id_split"),
                        }, dataset_type=DatasetType.IN_DISTRIBUTION, val_batch_size=self.classifier_cfg.val_batch_size,
            train_batch_size=self.classifier_cfg.train_batch_size,
            dataloader_num_workers=self.classifier_cfg.dataloader_num_workers,
            use_weighted_sampler=self.classifier_cfg.weighted_sampler)

        batch = next(iter(self.in_dist_train_dataloader))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
        assert batch['pixel_values'].shape == (self.classifier_cfg.train_batch_size, 3, self.resize, self.resize)
        assert batch['labels'].shape == (self.classifier_cfg.train_batch_size,)


    def train(self):
        self.initiate_trainer()
        if self.classifier_cfg.retrain:
            self.logger.info("Start training...")
            create_and_train_model(train_dataloader=self.in_dist_train_dataloader,
                                   val_dataloader=self.in_dist_val_dataloader,
                                   checkpoint=self.checkpoint_path,
                                   log_dir=self.train_output_dir,
                                   logger=self.logger,
                                   net_type=self.classifier_cfg.type,
                                   resume=self.classifier_cfg.resume,
                                   max_epoch=self.classifier_cfg.max_epoch,
                                   milestones=self.classifier_cfg.milestones,
                                   loss_class_weights=self.classifier_cfg.loss_class_weights)

    def get_fine_tuned_model(self):
        args = SimpleNamespace()
        args.net = self.classifier_cfg.type
        args.gpu = True
        args.num_classes = len(self.in_dist_train_dataloader.dataset.classes)
        model = get_network(args=args)
        checkpoints_folder = os.path.join(self.checkpoint_path, most_recent_folder(self.checkpoint_path, fmt=DATE_FORMAT))
        weights_file = best_acc_weights(checkpoints_folder)
        model.load_state_dict(torch.load(os.path.join(checkpoints_folder, weights_file)))
        return model


class VitClassifier(Classifier):
    def __init__(self, output_dir, classifier_cfg, logger, current_run_name, data_source, ood_class_names):
        super().__init__(output_dir, classifier_cfg, logger, current_run_name, data_source, ood_class_names)

    def get_vit_transforms(self):
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        image_mean = processor.image_mean
        image_std = processor.image_std
        size = processor.size["height"]
        # size=32
        normalize = Normalize(mean=image_mean, std=image_std)
        self.train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

        self.val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    def initiate_trainer(self):
        self.logger.info(f"Initializing classifier\n")
        self.preperae_id_ood_datasets(self.data_source, self.ood_class_names)
        self.get_vit_transforms()
        self.in_dist_train_dataloader, self.in_dist_val_dataloader, self.in_dist_test_dataloader = create_dataloaders(
            train_transforms=self.train_transforms, val_transforms=self.val_transforms,
            data_paths={"train": os.path.join(self.data_source, "train_ood_id_split"),
                        "val": os.path.join(self.data_source, "val_ood_id_split"),
                        "test": os.path.join(self.data_source, "test_ood_id_split"),
                        }, dataset_type=DatasetType.IN_DISTRIBUTION, val_batch_size=self.classifier_cfg.val_batch_size,
            train_batch_size=self.classifier_cfg.train_batch_size,
            dataloader_num_workers=self.classifier_cfg.dataloader_num_workers,
            use_weighted_sampler=self.classifier_cfg.weighted_sampler)

        self.id2label = self.in_dist_train_dataloader.dataset.labels_to_classes_names
        self.label2id = self.in_dist_train_dataloader.dataset.classes_names_to_labels

        self.model = ViTLightningModule(train_dataloader=self.in_dist_train_dataloader,
                                        val_dataloader=self.in_dist_val_dataloader,
                                        test_dataloader=self.in_dist_test_dataloader,
                                        id2label=self.id2label,
                                        label2id=self.label2id, num_labels=len(self.id2label),
                                        model_path=self.classifier_cfg.model_path,
                                        loss_class_weights=self.classifier_cfg.loss_class_weights, logger=self.logger,
                                        max_epochs=self.classifier_cfg.max_epoch)

        self.model = self.model.to(device)

    def train(self):
        self.initiate_trainer()
        if self.classifier_cfg.retrain:
            self.logger.info(f"Starting to train the ViT classifier\n")
            early_stop_callback = EarlyStopping(
                monitor='validation_loss',
                patience=3,
                strict=True,
                verbose=True,
                mode='min'
            )
            checkpoint_callback = ModelCheckpoint(
                monitor='validation_loss',
                mode='min',
                save_top_k=1,
                dirpath=self.checkpoint_path,
                filename='best_model',
            )

            logger = TensorBoardLogger(save_dir=os.path.join(self.train_output_dir, "tb_logs"), name=self.current_run_name)
            trainer = Trainer(num_nodes=1, max_epochs=self.classifier_cfg.max_epoch,
                              callbacks=[early_stop_callback, checkpoint_callback], logger=logger)
            ckpt_path = os.path.join(self.checkpoint_path, "best_model.ckpt") if self.classifier_cfg.resume else None
            trainer.fit(self.model, ckpt_path=ckpt_path)

            """Finally, let's test the trained model on the test set:"""

            results = trainer.test()
            print(results)

    def get_fine_tuned_model(self):
        model = ViTLightningModule.load_from_checkpoint(
            os.path.join(self.checkpoint_path, f"best_model.ckpt"),
            train_dataloader=self.in_dist_train_dataloader,
            val_dataloader=self.in_dist_val_dataloader,
            test_dataloader=self.in_dist_test_dataloader,
            id2label=self.id2label,
            label2id=self.label2id,
            num_labels=len(self.label2id)
        )
        return model
