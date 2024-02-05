import os
import pickle

import torch
import numpy as np
from enum import Enum

from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm

DEFAULT_ODIN_TEMP = 1000
DEFAULT_ODIN_NOISE_MAGNITUDE = 1e-3


class OODDatasetType(Enum):
    IN_DISTRIBUTION = 0
    OUT_OF_DISTRIBUTION = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OODDetector:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.train_output = os.path.join(self.output_dir, "train/OOD")
        self.test_output = os.path.join(self.output_dir, "test/OOD")
        os.makedirs(self.train_output, exist_ok=True)
        os.makedirs(self.test_output, exist_ok=True)
        self.test_dataset_scores_and_labels = os.path.join(self.test_output, "test_dataset_scores_and_labels.json")

    def score_samples(self, dataloader):
        pass

    def train(self):
        pass


class ODINOODDetector(OODDetector):
    def __init__(self, output_dir, epsilon=DEFAULT_ODIN_NOISE_MAGNITUDE,
                 temperature=DEFAULT_ODIN_TEMP, ):
        super().__init__(output_dir)
        self.model = None
        self.epsilon = epsilon
        self.temperature = temperature

    def calculate_odin_scores(self, inputs):
        criterion = nn.CrossEntropyLoss()
        inputs = Variable(inputs, requires_grad=True)
        inputs = inputs.cuda()
        inputs.retain_grad()

        outputs = self.model(inputs)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

        # Using temperature scaling
        outputs = outputs / self.temperature

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -self.epsilon, gradient)
        outputs = self.model(Variable(tempInputs))
        outputs = outputs / self.temperature
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
        scores = np.max(nnOutputs, axis=1)
        return torch.from_numpy(scores), torch.from_numpy(maxIndexTemp)

    def score_samples(self, dataloader):
        all_scores = []
        all_labels = []
        all_preds = []
        cache_dir = '/tmp/odin_scores/'
        os.makedirs(cache_dir, exist_ok=True)
        cache_index = 0
        for batch in tqdm(dataloader):
            images = batch['pixel_values']
            labels = batch['labels']
            scores, pred_labels = self.calculate_odin_scores(images.to(device))
            all_scores.append(scores)
            all_labels.append(labels)
            all_preds.append(pred_labels)
            if len(all_scores) > 1000:
                with open(os.path.join(cache_dir, f'all_scores_cache_{cache_index}.pkl'), 'wb') as f:
                    pickle.dump(all_scores, f)
                with open(os.path.join(cache_dir, f'all_labels_cache_{cache_index}.pkl'), 'wb') as f:
                    pickle.dump(all_labels, f)
                with open(os.path.join(cache_dir, f'all_preds_cache_{cache_index}.pkl'), 'wb') as f:
                    pickle.dump(all_preds, f)
                all_scores = []
                all_labels = []
                all_preds = []
                cache_index += 1
        # cache leftovers:
        with open(os.path.join(cache_dir, f'all_scores_cache_{cache_index}.pkl'), 'wb') as f:
            pickle.dump(all_scores, f)
        with open(os.path.join(cache_dir, f'all_labels_cache_{cache_index}.pkl'), 'wb') as f:
            pickle.dump(all_labels, f)
        with open(os.path.join(cache_dir, f'all_preds_cache_{cache_index}.pkl'), 'wb') as f:
            pickle.dump(all_preds, f)
        all_scores_caches = [x for x in os.listdir(cache_dir) if x.startswith('all_scores_cache_')]
        all_scores_caches.sort(key=lambda x: int(x.split('all_scores_cache_')[-1].split('.pkl')[0]))
        all_scores = []
        for cache_name in all_scores_caches:
            with open(os.path.join(cache_dir, cache_name), 'rb') as f:
                all_scores.append(pickle.load(f))
        all_scores = sum(all_scores, [])

        all_labels_caches = [x for x in os.listdir(cache_dir) if x.startswith('all_labels_cache_')]
        all_labels_caches.sort(key=lambda x: int(x.split('all_labels_cache_')[-1].split('.pkl')[0]))
        all_labels = []
        for cache_name in all_labels_caches:
            with open(os.path.join(cache_dir, cache_name), 'rb') as f:
                all_labels.append(pickle.load(f))
        all_labels = sum(all_labels, [])

        all_preds_caches = [x for x in os.listdir(cache_dir) if x.startswith('all_preds_cache_')]
        all_preds_caches.sort(key=lambda x: int(x.split('all_preds_cache_')[-1].split('.pkl')[0]))
        all_preds = []
        for cache_name in all_preds_caches:
            with open(os.path.join(cache_dir, cache_name), 'rb') as f:
                all_preds.append(pickle.load(f))
        all_preds = sum(all_preds, [])
        return torch.cat(all_scores, dim=0), torch.cat(all_labels, dim=0), torch.cat(all_preds, dim=0)
