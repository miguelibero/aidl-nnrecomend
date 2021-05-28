from statistics import mean
import math
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self, model: torch.nn.Module, trainloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn.Module, device: str=None, tb_dir: str=None):
        self.model = model
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.__tb = SummaryWriter(log_dir=tb_dir) if tb_dir else None

    def __call__(self, epoch: int=-1) -> float:
        """
        run a training epoch
        :return: the mean loss
        """
        total_loss = []
        self.model.train()

        for interactions in self.trainloader:
            self.optimizer.zero_grad()
            interactions = interactions.to(self.device)
            targets = interactions[:,2]
            predictions = self.model(interactions[:,:2])
            loss = self.criterion(predictions, targets.float())
            loss.backward()
            self.optimizer.step()
            total_loss.append(loss.item())

        total_loss = mean(total_loss)

        if self.__tb and epoch >= 0:
            self.__tb.add_scalar('train/loss', total_loss, epoch)

        return total_loss


class TestResult:
    def __init__(self, hr: float, ndcg: float, coverage: float):
        """
        :param hr: hit ratio
        :param ndcg: normalized discounted cumulative gain
        :param coverage: percentage of training data items recommended
        """ 
        self.hr = hr
        self.ndcg = ndcg 
        self.coverage = coverage


class Tester:

    def __init__(self, model: torch.nn.Module, testloader: torch.utils.data.DataLoader,
      trainloader: torch.utils.data.DataLoader, topk: int=10, device: str=None, tb_dir: str=None):
        self.model = model
        self.testloader = testloader
        self.topk = topk
        self.device = device
        self.__total_items = len(np.unique(trainloader.dataset[:,1]))
        self.__tb = SummaryWriter(log_dir=tb_dir) if tb_dir else None

    def __get_hit_ratio(self, ranking: torch.Tensor, item: torch.Tensor) -> int:
        """
        measures wether the test item is in the topk positions of the ranking
        """
        return 1 if item in ranking else 0

    def __get_ndcg(self, ranking: torch.Tensor, item: torch.Tensor) -> int:
        """
        normalized discounted cumulative gain
        measures the ranking quality with gives information about where in the ranking our test item is
        """
        idx = (ranking == item).nonzero(as_tuple=True)[0]
        return math.log(2)/math.log(idx[0]+2) if len(idx) > 0 else 0

    @torch.no_grad()
    def __call__(self, epoch: int = -1) -> TestResult:
        hr, ndcg = [], []

        total_recommended_items = set()

        for interactions in self.testloader:
            interactions = interactions[:,:2].to(self.device)
            real_item = interactions[0][1]
            predictions = self.model(interactions)
            _, indices = torch.topk(predictions, self.topk)
            recommended_items = interactions[indices][:, 1]
            total_recommended_items.update(recommended_items.tolist())
            hr.append(self.__get_hit_ratio(recommended_items, real_item))
            ndcg.append(self.__get_ndcg(recommended_items, real_item))

        cov = len(total_recommended_items) / self.__total_items
        result = TestResult(mean(hr), mean(ndcg), cov)

        if self.__tb and epoch >= 0:
            self.__tb.add_scalar('eval/HR@{topk}', result.hr, epoch)
            self.__tb.add_scalar('eval/NDCG@{topk}', result.ndcg, epoch)
            self.__tb.add_scalar('eval/COV@{topk}', result.coverage, epoch)

        return result