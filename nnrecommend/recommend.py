

import numpy as np
import statistics
import torch
from typing import Callable, Dict

from torch.utils.data.dataloader import DataLoader
from nnrecommend.dataset import InteractionDataset
from nnrecommend.operation import BaseSetup
from nnrecommend.hparams import HyperParameters


class Setup(BaseSetup):

    def __call__(self, hparams: HyperParameters) -> np.ndarray:

        hparams.pairwise_loss = None
        hparams.negatives_test = 0
        hparams.negatives_train = 0
        hparams.interaction_context = "previous"
        return super().__call__(hparams)

    def _load(self, hparams: HyperParameters) -> np.ndarray:
        super()._load(hparams)
        self.__apply(self.src.trainset)
        self.__apply(self.src.testset)
        return self.src.trainset.idrange

    def __apply(self, dataset: InteractionDataset) -> InteractionDataset:
        """
        prepare dataset to train for new users
        * set all users to value 0
        * move items to labels
        """
        dataset.remove_column(0) # remove users
        items = dataset[:, 0]
        dataset.remove_column(0) # remove items
        dataset[:, -1] = items # set items as labels
        return dataset


class Trainer:

    def __init__(self, model: torch.nn.Module, trainloader: DataLoader,
            optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: str=None):
        self.model = model
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def __call__(self) -> float:
        """
        run a training epoch
        :return: the mean loss
        """
        total_loss = []

        for batch in self.trainloader:
            self.optimizer.zero_grad()
            if self.device:
                batch = batch.to(self.device)
            interactions = batch[:,:-1]
            predictions = self.model(interactions)
            targets = batch[:,-1]
            loss = self.criterion(predictions, targets)
            loss.backward()
            self.optimizer.step()
            total_loss.append(loss.item())

        return statistics.mean(total_loss)


class TestResult:
    def __init__(self, accuracy: float):
        """
        :param accuracy: accuracy
        """ 
        self.accuracy = accuracy

    def to_dict(self) -> Dict[str, str]:
        return {
            "ACC": self.accuracy,
        }

    def __gt__(self, result: 'TestResult') -> bool:
        return self.accuracy > result.accuracy

    def __str__(self):
        return f"acc={self.accuracy:.4f}"


class Tester:
    def __init__(self, model: Callable, testloader: torch.utils.data.DataLoader,
      device: str=None):
        self.model = model
        self.testloader = testloader
        self.device = device

    @torch.no_grad()
    def __call__(self) -> TestResult:
        correct = 0
        total = 0
        for batch in self.testloader:
            if batch is None or batch.shape[0] == 0:
                continue
            if self.device:
                batch = batch.to(self.device)
            interactions = batch[:, :-1]
            targets = batch[:, -1]
            predict = self.model(interactions)
            _, predicted = torch.max(predict, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        return TestResult(correct/total)


class Model(torch.nn.Module):

    def __init__(self, idrange: np.ndarray, embed_dim: int):
        super().__init__()
        input_dim = len(idrange)
        # first col is previous items so output is that -1
        output_dim = idrange[0] - 1
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(input_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, output_dim),
        )
    
    def forward(self, x):
        x = x.float()
        return self.linear(x)

def create_model(hparams: HyperParameters, idrange: np.ndarray) -> torch.nn.Module:
    return Model(idrange, hparams.embed_dim)


def create_model_training(model: torch.nn.Module, hparams: HyperParameters):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        patience=hparams.lr_scheduler_patience,
        factor=hparams.lr_scheduler_factor,
        threshold=hparams.lr_scheduler_threshold)
    return criterion, optimizer, scheduler