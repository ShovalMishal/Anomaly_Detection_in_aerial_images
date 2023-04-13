import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from transformers import ViTImageProcessor
from torch.utils.data import DataLoader
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)
from datasets import load_dataset
from PIL import ImageDraw, ImageFont, Image
from vit_model import ViTLightningModule


def show_examples(ds, seed: int = 1234, examples_per_class: int = 3, size=(350, 350)):
    w, h = size
    labels = ds.features['label'].names
    grid = Image.new('RGB', size=(examples_per_class * w, len(labels) * h))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", 24)

    for label_id, label in enumerate(labels):

        # Filter the dataset by a single label, shuffle it, and grab a few samples
        ds_slice = ds.filter(lambda ex: ex['label'] == label_id).shuffle(seed).select(
            range(examples_per_class))

        # Plot this label's examples along a row
        for i, example in enumerate(ds_slice):
            image = example['image']
            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class * w, idx // examples_per_class * h)
            grid.paste(image.resize(size), box=box)
            draw.text(box, label, (255, 255, 255), font=font)
    grid.show()
    return grid

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples


def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help="the relative path to save the output database")
    parser.add_argument("-trb", "--train_batch_size", help="The train batch size to be set")
    parser.add_argument("-teb", "--eval_batch_size", help="The test batch size to be set")

    args = parser.parse_args()
    train_path = os.path.join(args.path, "train", "**")
    test_path = os.path.join(args.path, "test", "**")
    # loading dataset
    bb_dataset = load_dataset("imagefolder", data_files={"train": train_path, "test": test_path})
    train_ds = bb_dataset['train']
    test_ds = bb_dataset['test']
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']
    # loading model
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
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

    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)
    test_ds.set_transform(val_transforms)
    id2label = {id: label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label: id for id, label in id2label.items()}
    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size)
    val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=args.eval_batch_size)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )
    labels = train_ds.features['label'].names
    model = ViTLightningModule(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                               test_dataloader=test_dataloader, id2label=id2label, label2id=label2id,
                               trainnum_labels=len(labels))
    trainer = Trainer(accelerator="gpu", num_nodes=1, callbacks=[EarlyStopping(monitor='validation_loss')])
    trainer.fit(model)
