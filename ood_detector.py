import torch
import numpy as np
from enum import Enum
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_ODIN_TEMP = 1000
DEFAULT_ODIN_NOISE_MAGNITUDE = 1e-3


class OODDatasetType(Enum):
    IN_DISTRIBUTION = 0
    OUT_OF_DISTRIBUTION = 1


class OODDetector:
    def __init__(self, model, id_dataloader, ood_dataloader):
        self.model = model
        self.id_dataloader = id_dataloader
        self.ood_dataloader = ood_dataloader

    def score_samples(self, dataset_type: OODDatasetType = OODDatasetType.IN_DISTRIBUTION):
        pass


class ODINOODDetector(OODDetector):
    def __init__(self, model, id_dataloader, ood_dataloader, epsilon=DEFAULT_ODIN_NOISE_MAGNITUDE,
                 temperature=DEFAULT_ODIN_TEMP,):
        super().__init__(model, id_dataloader, ood_dataloader)
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
        return torch.from_numpy(scores)

    def score_samples(self, dataset_type: OODDatasetType = OODDatasetType.IN_DISTRIBUTION):
        data_loader = {OODDatasetType.IN_DISTRIBUTION: self.id_dataloader,
                       OODDatasetType.OUT_OF_DISTRIBUTION: self.ood_dataloader}[dataset_type]
        all_scores = []
        for batch in data_loader:
            images, labels = batch[0], batch[1]
            scores = self.calculate_odin_scores(images)
            all_scores.append(scores)
        return torch.cat(all_scores)

