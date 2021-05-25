from statistics import mean
import math
import torch
import numpy as np

class Trainer:

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

    def __init__(self, model: torch.nn.Module, trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
            criterion: torch.nn.Module, device: str):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.__total_items = len(np.unique(self.trainloader.dataset[:,1]))

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

    def __call__(self) -> float:
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

        return mean(total_loss)

    @torch.no_grad()
    def test(self, topk: int=10) -> TestResult:
        """
        Test the HR and NDCG for the model @topK
        """
        self.model.eval()
        hr, ndcg = [], []

        total_recommended_items = set()

        for interactions in self.testloader:
            interactions = interactions[:,:2].to(self.device)
            real_item = interactions[0][1]
            predictions = self.model.forward(interactions)
            _, indices = torch.topk(predictions, topk)
            recommended_items = interactions[indices][:, 1]
            total_recommended_items.update(recommended_items.tolist())
            hr.append(self.__get_hit_ratio(recommended_items, real_item))
            ndcg.append(self.__get_ndcg(recommended_items, real_item))

        cov = len(total_recommended_items) / self.__total_items
        return self.TestResult(mean(hr), mean(ndcg), cov)