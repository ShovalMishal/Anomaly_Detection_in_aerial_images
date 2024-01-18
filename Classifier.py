import os
from typing import Union, Optional, Dict, Callable, List, Tuple

import torch

from plotly.figure_factory import np
from torch import nn
from torch.utils.data import WeightedRandomSampler

from utils import create_patches_dataset
from datasets import load_metric, Dataset
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer, PreTrainedModel, \
    DataCollator, PreTrainedTokenizerBase, EvalPrediction, TrainerCallback
from CustomSampler import CustomSampler


class Classifier():
    def __init__(self, output_dir, classifier_cfg, logger):
        self.classifier_cfg = classifier_cfg
        self.classfier_train_output_dir = os.path.join(output_dir, classifier_cfg.train_output_dir)
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
    def __init__(self, output_dir, classifier_cfg, logger):
        super().__init__(output_dir, classifier_cfg, logger)
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
                                                                    logger=self.logger,
                                                                    is_valid_file_use=True,
                                                                    ood_remove=True)
        self.in_distribution_val_dataset = create_patches_dataset(type="subval", dataset_cfg=dataset_cfg,
                                                                  transform=self.vit_transform, output_dir=output_dir,
                                                                  logger=self.logger,
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
            output_dir=self.classfier_train_output_dir,
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
        if self.classifier_cfg.sampler_type == "random":
            self.trainer = Trainer(model=model,
                                   args=training_args,
                                   data_collator=self.collate_fn,
                                   compute_metrics=compute_metrics,
                                   train_dataset=self.in_distribution_train_dataset,
                                   eval_dataset=self.in_distribution_val_dataset,
                                   tokenizer=self.feature_extractor, )
        else:
            self.trainer = CustomTrainer(
                sampler_type=self.classifier_cfg.sampler_type,
                sampler_cfg=self.classifier_cfg.sampler_cfg,
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
        self.trainer.log_metrics("train", train_results.metrics)
        self.trainer.save_metrics("train", train_results.metrics)
        self.trainer.save_state()
        # evaluate
        metrics = self.trainer.evaluate(self.in_distribution_val_dataset)
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

    def get_fine_tuned_model(self):
        model = ViTForImageClassification.from_pretrained(self.classfier_train_output_dir)
        return model
