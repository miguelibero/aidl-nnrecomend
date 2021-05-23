from statistics import mean
import math
import torch


class Trainer:

    class TestResult:
        def __init__(self, hr: float, ndcg: float):
            """
            :param hr: hit ratio
            :param ndcg: normalized discounted cumulative gain
            """ 
            self.hr = hr
            self.ndcg = ndcg 

    def __init__(self, model: torch.nn.Module, trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
            criterion: torch.nn.Module, device: str):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

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
            interactions = interactions.to(self.device)
            targets = interactions[:,2]
            predictions = self.model(interactions[:,:2])
            loss = self.criterion(predictions, targets.float())
            self.model.zero_grad()
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

        for interactions in self.testloader:
            interactions = interactions[:,:2].to(self.device)
            good_item = interactions[0][1]
            predictions = self.model.forward(interactions)
            _, indices = torch.topk(predictions, topk)
            recommended_items = interactions[indices][:, 1]
            hr.append(self.__get_hit_ratio(recommended_items, good_item))
            ndcg.append(self.__get_ndcg(recommended_items, good_item))
        return self.TestResult(mean(hr), mean(ndcg))