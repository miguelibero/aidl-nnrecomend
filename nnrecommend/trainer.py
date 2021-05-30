from statistics import mean
import math
import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, model: torch.nn.Module, trainloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: str=None,
            tb_dir: str=None, tb_tag: str=None):
        self.model = model
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.__setup_tensorboard(self.trainloader.dataset, tb_dir, tb_tag)

            
    def __setup_tensorboard(self, data: np.ndarray, tb_dir: str, tb_tag: str):
        self.__tb = create_tensorboard_writer(tb_dir, tb_tag)
        if not self.__tb:
            return
        self.__tb_tag = tb_tag
        maxuser = np.max(data[:,0]).astype(int)
        maxitem = np.max(data[:,1]).astype(int)
        embsize = maxitem + 1
        self.__tb_labels = []
        self.__tb_imgs = np.zeros((embsize, 3, 1, 1))
        for i in range(embsize):
            if i <= maxuser:
                label = f"u{i}"
                color = np.array((1, 0, 0))
            else:
                label = f"i{i}"
                color = np.array((0, 0, 1))
            self.__tb_labels.append(label)
            self.__tb_imgs[i, :, 0, 0] = color
        self.__tb_imgs = torch.from_numpy(self.__tb_imgs)


    def __call__(self, epoch: int=-1) -> float:
        """
        run a training epoch
        :return: the mean loss
        """
        total_loss = []
        self.model.train()

        for rows in self.trainloader:
            self.optimizer.zero_grad()
            rows = rows.to(self.device)
            interactions = rows[:,:2].long()
            targets = rows[:,2].float()
            predictions = self.model(interactions)
            loss = self.criterion(predictions, targets)
            loss.backward()
            self.optimizer.step()
            total_loss.append(loss.item())

        total_loss = mean(total_loss)
        self.__update_tensorboard(total_loss, epoch)

        return total_loss

    def __update_tensorboard(self, loss: int, epoch: int):
        if not self.__tb or epoch < 0:
            return
        self.__tb.add_scalar('train/loss', loss, epoch)
        if hasattr(self.model, 'get_embedding_weight'):
            weight = self.model.get_embedding_weight()
            self.__tb.add_embedding(weight, global_step=epoch, tag=self.__tb_tag,
                metadata=self.__tb_labels, label_img=self.__tb_imgs)


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
      trainloader: torch.utils.data.DataLoader, topk: int=10, device: str=None,
      tb_dir: str=None, tb_tag: str=None):
        self.model = model
        self.testloader = testloader
        self.topk = topk
        self.device = device
        self.__total_items = len(np.unique(trainloader.dataset[:,1]))
        self.__tb = create_tensorboard_writer(tb_dir, tb_tag)

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

        for rows in self.testloader:
            rows = rows.to(self.device)
            interactions = rows[:,:2].long()
            real_item = interactions[0][1]
            predictions = self.model(interactions)
            _, indices = torch.topk(predictions, self.topk)
            recommended_items = interactions[indices][:, 1]
            total_recommended_items.update(recommended_items.tolist())
            hr.append(self.__get_hit_ratio(recommended_items, real_item))
            ndcg.append(self.__get_ndcg(recommended_items, real_item))

        cov = len(total_recommended_items) / self.__total_items
        result = TestResult(mean(hr), mean(ndcg), cov)

        if self.__tb is not None and epoch >= 0:
            self.__tb.add_scalar(f'eval/HR@{self.topk}', result.hr, epoch)
            self.__tb.add_scalar(f'eval/NDCG@{self.topk}', result.ndcg, epoch)
            self.__tb.add_scalar(f'eval/COV@{self.topk}', result.coverage, epoch)

        return result

def create_tensorboard_writer(tb_dir, tb_tag):
    if not tb_dir:
        return
    if tb_tag:
        tb_dir = os.path.join(tb_dir, tb_tag)
    return SummaryWriter(log_dir=tb_dir)