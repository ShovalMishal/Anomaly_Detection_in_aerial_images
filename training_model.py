import os
from argparse import ArgumentParser
import numpy as np
import torch
from datasets import load_dataset, load_metric
from PIL import ImageDraw, ImageFont, Image


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


def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help="The relative path to the input database")
    parser.add_argument('-id', '--in_distribution_list', nargs='+', default=[])

    args = parser.parse_args()
    # train on in distribution classes
    train_paths = [os.path.join(args.path, "train", label, "**") for label in args.in_distribution_list]
    test_path = [os.path.join(args.path, "test", label, "**") for label in args.in_distribution_list]
    # loading datasets
    bb_dataset = load_dataset("imagefolder", data_files={"train": train_paths, "test": test_path})
    train_ds = bb_dataset['train']
    test_ds = bb_dataset['test']
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']
    # loading model
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)
    train_ds.set_transform(transform=transform)
    test_ds.set_transform(transform=transform)
    val_ds.set_transform(transform=transform)

    metric = load_metric("accuracy")
    labels = train_ds.features['label'].names
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )
    training_args = TrainingArguments(
        output_dir="./vit-bb-dataset44",
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=4,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=feature_extractor,
    )

    # train model
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # evaluate
    metrics = trainer.evaluate(test_ds)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
