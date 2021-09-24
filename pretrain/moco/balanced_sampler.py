import random
import torch
import math
import torch.utils.data
import torch.distributed as dist

class BalancedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.dataset_dict = {0: [], 1: []}
        self.balanced_max = 0

        # Save all the indices for all the classes
        labels = dataset.get_label()
        for idx in range(0, len(labels)):
            label = labels[idx]
            self.dataset_dict[label].append(idx)

        # Oversample the classes with fewer elements than the max
        self.balanced_max = max(len(self.dataset_dict[0]), len(self.dataset_dict[1]))

        while len(self.dataset_dict[0]) < self.balanced_max:
            self.dataset_dict[0].append(random.choice(self.dataset_dict[0]))

        while len(self.dataset_dict[1]) < self.balanced_max:
            self.dataset_dict[1].append(random.choice(self.dataset_dict[1]))


    def __iter__(self):
        dataset_dict = {0: list(self.dataset_dict[0]), 1: list(self.dataset_dict[1])}
        # shuffle
        random.shuffle(dataset_dict[0])
        random.shuffle(dataset_dict[1])

        result = [None]*(self.balanced_max*2)
        result[::2] = dataset_dict[0]
        result[1::2] = dataset_dict[1]
        return iter(result)
    
    def __len__(self):
        return self.balanced_max * 2


class DistributedBalancedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()

        self.dataset_dict = {0: [], 1: []}
        labels = dataset.get_label()
        for idx in range(0, len(labels)):
            label = labels[idx]
            self.dataset_dict[label].append(idx)

        # Oversample the classes with fewer elements than the max
        self.balanced_max = max(len(self.dataset_dict[0]), len(self.dataset_dict[1]))

        while len(self.dataset_dict[0]) < self.balanced_max:
            self.dataset_dict[0].append(random.choice(self.dataset_dict[0]))

        while len(self.dataset_dict[1]) < self.balanced_max:
            self.dataset_dict[1].append(random.choice(self.dataset_dict[1]))

        if (self.balanced_max) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (self.balanced_max - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(self.balanced_max / self.num_replicas)  # type: ignore

        self.total_size = self.num_samples * self.num_replicas


    def __iter__(self):
        # shuffle
        random.seed(self.epoch)
        dataset_dict = {0: list(self.dataset_dict[0]), 1: list(self.dataset_dict[1])}
        random.shuffle(dataset_dict[0])
        random.shuffle(dataset_dict[1])

        result = [None]*(self.num_samples*2)
        result[::2] = dataset_dict[0][self.rank:self.total_size:self.num_replicas]
        result[1::2] = dataset_dict[1][self.rank:self.total_size:self.num_replicas]
        return iter(result)
    
    def __len__(self):
        return self.num_samples*2

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch