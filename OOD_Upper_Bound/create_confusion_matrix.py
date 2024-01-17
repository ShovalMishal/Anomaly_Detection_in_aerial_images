# !pip install -q transformers datasets pytorch-lightning

"""## Load the data

Here we import a small portion of CIFAR-10, for demonstration purposes. This dataset can be found on the [hub](https://huggingface.co/datasets/cifar10) (you can view images directly in your browser!).
"""

from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from torchvision.datasets import ImageFolder
import matplotlib as plt
from utils import eval_model

"""## Preprocessing the data

We will now preprocess the data. The model requires 2 things: `pixel_values` and `labels`.

We will perform data augmentaton **on-the-fly** using HuggingFace Datasets' `set_transform` method (docs can be found [here](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=set_transform#datasets.Dataset.set_transform)). This method is kind of a lazy `map`: the transform is only applied when examples are accessed. This is convenient for tokenizing or padding text, or augmenting images at training time for example, as we will do here.

We first load the image processor, which is a minimal object that can be used to prepare images for inference. We use some of its properties which are relevant for preparing images for the model.
"""

from transformers import ViTImageProcessor, ViTForImageClassification


processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size["height"]

"""For data augmentation, one can use any available library. Here we'll use torchvision's [transforms module](https://pytorch.org/vision/stable/transforms.html)."""

from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

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

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

# load cifar10 (only small portion for demonstration purposes)
# train_ds, test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:2000]'])
train_ds = ImageFolder(root='/home/shoval/Documents/Repositories/data/DOTAV2_fg_patches/train/images', transform=_train_transforms)
val_ds = ImageFolder(root='/home/shoval/Documents/Repositories/data/DOTAV2_fg_patches/val/images', transform=_val_transforms)
# split up training into training + validation

"""We can also check out the features of the dataset in more detail:"""

"""As we can see, each example has 2 features: 'img' (of type `Image`) and 'label' (of type `ClassLabel`). Let's check an example of the training dataset:"""


"""Of course, we would like to know the actual class name, rather than the integer index. We can obtain that by creating a dictionary which maps between integer indices and actual class names (id2label):"""

id2label = {idx: class_name for idx, class_name in enumerate(train_ds.classes)}
label2id = {class_name: idx for idx, class_name in enumerate(train_ds.classes)}

"""We can now load preprocessed images (on-the-fly) as follows:"""

"""It's very easy to create corresponding PyTorch DataLoaders, like so:"""

from torch.utils.data import DataLoader
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

train_batch_size = 128
eval_batch_size = 128

train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size, num_workers=12)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=12)

batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k, v.shape)

assert batch['pixel_values'].shape == (train_batch_size, 3, 224, 224)
assert batch['labels'].shape == (train_batch_size,)

next(iter(val_dataloader))['pixel_values'].shape

from our_google_finetuner import ViTLightningModule
train_ds = ImageFolder(root='/home/shoval/Documents/Repositories/data/DOTAV2_fg_patches/train/images',
                           transform=_train_transforms)
val_ds = ImageFolder(root='/home/shoval/Documents/Repositories/data/DOTAV2_fg_patches/val/images',
                     transform=_val_transforms)
# split up training into training + validation

"""We can also check out the features of the dataset in more detail:"""

"""As we can see, each example has 2 features: 'img' (of type `Image`) and 'label' (of type `ClassLabel`). Let's check an example of the training dataset:"""

"""Of course, we would like to know the actual class name, rather than the integer index. We can obtain that by creating a dictionary which maps between integer indices and actual class names (id2label):"""

id2label = {idx: class_name for idx, class_name in enumerate(train_ds.classes)}
label2id = {class_name: idx for idx, class_name in enumerate(train_ds.classes)}

train_batch_size = 128
eval_batch_size = 128

train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size,
                              num_workers=12)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=12)


model = ViTLightningModule(train_dataloader, val_dataloader, test_dataloader=val_dataloader, id2label=id2label, label2id=label2id, num_labels=18,)
checkpoint_path = '/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/lightning_logs/version_5/checkpoints/epoch=0-step=2098.ckpt'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()
all_preds, all_labels = eval_model(dataloader=val_dataloader, model=model)
res_confusion_matrix = confusion_matrix(all_labels, all_preds)
plt.imshow(res_confusion_matrix); plt.show()
import matplotlib.pyplot as plt
res_confusion_matrix = confusion_matrix(all_labels, all_preds, labels=train_ds.classes)
plt.imshow(res_confusion_matrix); plt.xticks(range(len(train_ds.classes)), train_ds.classes, rotation=90); plt.yticks(range(len(train_ds.classes)), train_ds.classes); plt.tight_layout(); plt.show()
res_confusion_matrix.sum()
correct=0
for i in range(len(res_confusion_matrix)): correct += res_confusion_matrix[i, i]
correct / res_confusion_matrix.sum()
correct / res_confusion_matrix.sum() * 100
confusion_mat_acc = res_confusion_matrix.copy()

confusion_mat_acc = res_confusion_matrix.copy().astype(float)
for idx in range(len(confusion_mat_acc)):
    confusion_mat_acc[idx,...] = confusion_mat_acc[idx,...] / confusion_mat_acc[idx, ...].sum()
plt.imshow(confusion_mat_acc); plt.colorbar(); plt.xticks(range(len(train_ds.classes)), train_ds.classes, rotation=90); plt.yticks(range(len(train_ds.classes)), train_ds.classes); plt.title(f'confusion matrix normalized according to rows\naccuracy: {correct / res_confusion_matrix.sum() * 100:.2f}[%]'); plt.tight_layout(); plt.show()
