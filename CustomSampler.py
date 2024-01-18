import random
from typing import Optional, Sized

from torch.utils.data import Sampler


class CustomSampler(Sampler):
    def __init__(self, labels, num_samples, labels_frequency_inbatch, batch_size, shuffle=True):
        self.labels = labels
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.labels_frequency_inbatch = labels_frequency_inbatch
        self.batch_size = batch_size
        self.batches_num = self.num_samples // batch_size
        # Create a dictionary to store the indices of the samples for each label
        self.label_indices = {}
        for label in set(labels):
            self.label_indices[label] = []

        # Iterate over the samples and add them to the appropriate label index list
        for i, label in enumerate(labels):
            self.label_indices[label].append(i)

        # Shuffle the label indices if necessary
        if shuffle:
            for label, indices in self.label_indices.items():
                random.shuffle(indices)

        self.indices = []

    def __iter__(self):
        for i in range(self.batches_num):
            curr_batch = []
            for label, indices in self.label_indices.items():
                curr_batch += indices[int(i * self.labels_frequency_inbatch[label]) % len(indices):int(i + 1) *
                                                                                                   self.labels_frequency_inbatch[
                                                                                                       label] % len(
                    indices)]
            self.indices += curr_batch
        # Return the list of sample indices
        return iter(self.indices)

    def __len__(self):
        # Return the total number of samples
        return self.num_samples
