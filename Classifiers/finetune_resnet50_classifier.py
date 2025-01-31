import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AdamW, get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNet50LightningModule(pl.LightningModule):
    def __init__(self, train_dataloader, val_dataloader, test_dataloader,
                 loss_class_weights=True, num_labels=10, max_epochs=15):
        super(ResNet50LightningModule, self).__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        self.training_weights = torch.tensor(list(self._train_dataloader.dataset.class_weights.values())).to(device) \
            if loss_class_weights else None
        self.validation_weights = torch.tensor(list(self._val_dataloader.dataset.class_weights.values())).to(device) \
            if loss_class_weights else None
        self.testing_weights = torch.tensor(list(self._test_dataloader.dataset.class_weights.values())).to(device) \
            if loss_class_weights else None
        # Load pre-trained ResNet-50
        self.model = models.resnet50(pretrained=True)
        # Replace the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_labels)
        self.model = self.model.to(device)
        self.num_training_steps = len(self._train_dataloader) * max_epochs
        self.num_warmup_steps = self.num_training_steps // 10
        self.criterions = {"train": nn.CrossEntropyLoss(weight=self.training_weights),
                           "val": nn.CrossEntropyLoss(weight=self.validation_weights),
                           "test": nn.CrossEntropyLoss(weight=self.testing_weights)}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        outputs = self(images)
        loss = self.criterions["train"](outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("training_loss", loss)
        self.log('training_accuracy', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        outputs = self(images)
        loss = self.criterions["val"](outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('validation_loss', loss, prog_bar=True)
        self.log('validation_accuracy', acc, prog_bar=True)


    def test_step(self, batch, batch_idx):
        images = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        outputs = self(images)
        loss = self.criterions["test"](outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('testing_loss', loss, prog_bar=True)
        self.log('testing_accuracy', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5, weight_decay=0.001)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Called after every batch
                'frequency': 1
            }
        }

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def pen_ultimate_layer(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x