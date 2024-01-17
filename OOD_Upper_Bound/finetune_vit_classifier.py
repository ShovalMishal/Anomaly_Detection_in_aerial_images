# !pip install -q transformers datasets pytorch-lightning

"""## Load the data

Here we import a small portion of CIFAR-10, for demonstration purposes. This dataset can be found on the [hub](https://huggingface.co/datasets/cifar10) (you can view images directly in your browser!).
"""
import json
import os
from enum import Enum

from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from torchvision.datasets import ImageFolder

from OODDetector import ODINOODDetector
from OOD_Upper_Bound.ood_and_id_dataset import OODAndIDDataset
from results import plot_confusion_matrix
from utils import eval_model

"""## Preprocessing the data

We will now preprocess the data. The model requires 2 things: `pixel_values` and `labels`.

We will perform data augmentaton **on-the-fly** using HuggingFace Datasets' `set_transform` method (docs can be found [here](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=set_transform#datasets.Dataset.set_transform)). This method is kind of a lazy `map`: the transform is only applied when examples are accessed. This is convenient for tokenizing or padding text, or augmenting images at training time for example, as we will do here.

We first load the image processor, which is a minimal object that can be used to prepare images for inference. We use some of its properties which are relevant for preparing images for the model.
"""

from transformers import ViTImageProcessor

"""For data augmentation, one can use any available library. Here we'll use torchvision's [transforms module](https://pytorch.org/vision/stable/transforms.html)."""

from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

"""We can now load preprocessed images (on-the-fly) as follows:"""

"""It's very easy to create corresponding PyTorch DataLoaders, like so:"""

from torch.utils.data import DataLoader
import torch


def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


"""## Define the model

Here we define a `LightningModule`, which is very similar to a regular `nn.Module`, but with some additional functionalities.

The model itself uses a linear layer on top of a pre-trained `ViTModel`. We place a linear layer on top of the last hidden state of the [CLS] token, which serves as a good representation of an entire image. We also add dropout for regularization.

A resource that helped me in understanding PyTorch Lightning is the [documentation](https://pytorch-lightning.readthedocs.io/en/latest/index.html) as well as the [tutorial notebooks](https://github.com/PyTorchLightning/pytorch-lightning/tree/master/notebooks).
"""

import pytorch_lightning as pl
from transformers import ViTForImageClassification, AdamW
import torch.nn as nn


class ViTLightningModule(pl.LightningModule):
    def __init__(self, train_dataloader, val_dataloader, test_dataloader, id2label, label2id, num_labels=10, ):
        super(ViTLightningModule, self).__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                             num_labels=num_labels,
                                                             id2label=id2label,
                                                             label2id=label2id)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct / pixel_values.shape[0]

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=5e-5)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader


"""## Train the model

Let's first start up Tensorboard (note that PyTorch Lightning logs to Tensorboard by default):
"""

# Commented out IPython magic to ensure Python compatibility.
# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/

"""Let's initialize the model, and train it!

We also add a callback:
* early stopping (stop training once the validation loss stops improving after 3 times).
"""

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping


class DatasetType(Enum):
    IN_DISTRIBUTION = "id"
    OUT_OF_DISTRIBUTION = "ood"
    NONE = ""


# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
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


def train_classifier(train_dataloader, val_dataloader, num_labels=18, train=True):
    # train_ds, test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:2000]'])

    # split up training into training + validation

    """We can also check out the features of the dataset in more detail:"""

    """As we can see, each example has 2 features: 'img' (of type `Image`) and 'label' (of type `ClassLabel`). Let's check an example of the training dataset:"""

    """Of course, we would like to know the actual class name, rather than the integer index. We can obtain that by creating a dictionary which maps between integer indices and actual class names (id2label):"""

    id2label = {idx: class_name for idx, class_name in enumerate(train_dataloader.dataset.classes)}
    label2id = {class_name: idx for idx, class_name in enumerate(train_dataloader.dataset.classes)}

    train_batch_size = 100
    eval_batch_size = 100

    batch = next(iter(train_dataloader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)

    assert batch['pixel_values'].shape == (train_batch_size, 3, 224, 224)
    assert batch['labels'].shape == (train_batch_size,)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )
    model = ViTLightningModule(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                               test_dataloader=val_dataloader,
                               id2label=id2label,
                               label2id=label2id, num_labels=num_labels, )

    if train:
        trainer = Trainer(num_nodes=1, callbacks=[EarlyStopping(monitor='validation_loss')])
        trainer.fit(model)

        """Finally, let's test the trained model on the test set:"""

        trainer.test()
        model = trainer.model
    else:
        model = ViTLightningModule.load_from_checkpoint(
            "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/OOD_Upper_Bound/lightning_logs/version_0/checkpoints/epoch=7-step=14248.ckpt",
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=val_dataloader,
            id2label=id2label,
            label2id=label2id,
            num_labels=len(train_dataloader.dataset.classes)
        )
    return model
