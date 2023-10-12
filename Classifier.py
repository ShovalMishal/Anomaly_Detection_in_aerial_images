import os
import torch
from plotly.figure_factory import np

from utils import create_patches_dataset
from datasets import load_metric
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer


class Classifier():
    def __init__(self, output_dir, classifier_cfg):
        self.classfier_output_dir = os.path.join(output_dir, classifier_cfg.output_dir)

    def train(self):
        pass

    def test(self):
        pass


# class Tokenizer(PreTrainedTokenizerBase)
def create_transform_func(feature_extractor):
    def vit_transform(example_batch):
        # Take a list of PIL images and turn them into pixel values
        inputs = feature_extractor(example_batch, return_tensors='pt')
        return inputs

    return vit_transform


def compute_metrics(p):
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


class VitClassifier(Classifier):
    def __init__(self, output_dir, classifier_cfg):
        super().__init__(output_dir, classifier_cfg)
        self.feature_extractor = ViTImageProcessor.from_pretrained(classifier_cfg["model_path"])
        self.vit_transform = create_transform_func(self.feature_extractor)

    @staticmethod
    def collate_fn(examples):
        pixel_values = torch.cat([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def initiate_trainer(self, classifier_cfg, dataset_cfg, output_dir):
        self.in_distribution_train_dataset = create_patches_dataset(type="subtrain", dataset_cfg=dataset_cfg,
                                                                    transform=self.vit_transform,
                                                                    output_dir=output_dir,
                                                                    is_valid_file_use=True,
                                                                    ood_remove=True)
        self.in_distribution_val_dataset = create_patches_dataset(type="subval", dataset_cfg=dataset_cfg,
                                                                  transform=self.vit_transform, output_dir=output_dir,
                                                                  is_valid_file_use=True,
                                                                  ood_remove=True)
        labels_to_classe_names = self.in_distribution_train_dataset.labels_to_classe_names
        classes_names_to_labels = self.in_distribution_train_dataset.classes_names_to_labels
        model = ViTForImageClassification.from_pretrained(
            classifier_cfg.model_path,
            num_labels=len(labels_to_classe_names),
            id2label=labels_to_classe_names,
            label2id=classes_names_to_labels
        )
        training_args = TrainingArguments(
            output_dir=self.classfier_output_dir,
            per_device_train_batch_size=classifier_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=classifier_cfg.per_device_eval_batch_size,
            evaluation_strategy=classifier_cfg.evaluation_strategy,
            num_train_epochs=classifier_cfg.num_train_epochs,
            fp16=classifier_cfg.fp16,
            save_steps=classifier_cfg.save_steps,
            eval_steps=classifier_cfg.eval_steps,
            logging_steps=classifier_cfg.logging_steps,
            learning_rate=classifier_cfg.learning_rate,
            save_total_limit=classifier_cfg.save_total_limit,
            remove_unused_columns=classifier_cfg.remove_unused_columns,
            push_to_hub=classifier_cfg.push_to_hub,
            report_to=classifier_cfg.report_to,
            load_best_model_at_end=classifier_cfg.load_best_model_at_end,
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=self.collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=self.in_distribution_train_dataset,
            eval_dataset=self.in_distribution_val_dataset,
            tokenizer=self.feature_extractor,
        )

    def train(self):
        train_results = self.trainer.train()
        self.trainer.save_model()
        self.trainer.log_metrics("subtrain", train_results.metrics)
        self.trainer.save_metrics("subtrain", train_results.metrics)
        self.trainer.save_state()
        # evaluate
        metrics = self.trainer.evaluate(self.in_distribution_val_dataset)
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

    def get_fine_tuned_model(self):
        model = ViTForImageClassification.from_pretrained(self.classfier_output_dir)
        return model
