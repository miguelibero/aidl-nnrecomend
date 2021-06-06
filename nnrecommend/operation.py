from logging import Logger
from nnrecommend.hparams import HyperParameters
from typing import Dict
from torch.utils.data.dataloader import DataLoader
from nnrecommend.logging import get_logger
from nnrecommend.dataset import BaseDatasetSource
from statistics import mean
import math
import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter


class Setup:

    def __init__(self, src: BaseDatasetSource, logger: Logger=None):
        self.src = src
        self.__logger = logger or get_logger(self)

    def __call__(self, hparams: HyperParameters):
        self.__logger.info("loading dataset...")

        self.src.load(hparams.max_interactions)
        idrange = self.src.trainset.idrange
        maxids = idrange - 1
        maxids[1] -= maxids[0]
        trainlen = len(self.src.trainset)
        testlen = len(self.src.testset)
        self.__logger.info(f"loaded {trainlen}/{testlen} interactions of {maxids[0]} users and {maxids[1]} items")

        self.__logger.info("adding negative sampling...")
        matrix = self.src.matrix
        self.src.trainset.add_negative_sampling(matrix, hparams.negatives_train)
        self.src.testset.add_negative_sampling(matrix, hparams.negatives_test)

        return idrange

    def create_testloader(self, hparams: HyperParameters):
        # test loader should not be shuffled since the negative samples need to be consecutive
        return DataLoader(self.src.testset, batch_size=hparams.negatives_test+1, num_workers=0)

    def create_trainloader(self, hparams: HyperParameters):
        return DataLoader(self.src.trainset, batch_size=hparams.batch_size, shuffle=True, num_workers=0)


class Trainer:

    def __init__(self, model: torch.nn.Module, trainloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: str=None):
        self.model = model
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def __call__(self, epoch: int=None) -> float:
        """
        run a training epoch
        :return: the mean loss
        """
        total_loss = []
        self.model.train()

        for rows in self.trainloader:
            self.optimizer.zero_grad()
            if self.device:
                rows = rows.to(self.device)
            interactions = rows[:,:2].long()
            targets = rows[:,2].float()
            predictions = self.model(interactions)
            loss = self.criterion(predictions, targets)
            loss.backward()
            self.optimizer.step()
            total_loss.append(loss.item())

        return mean(total_loss)


class TestResult:
    def __init__(self, topk: int, hr: float, ndcg: float, coverage: float):
        """
        :param topk: number of topk used to obtain the vaues
        :param hr: hit ratio
        :param ndcg: normalized discounted cumulative gain
        :param coverage: percentage of training data items recommended
        """ 
        self.topk = topk
        self.hr = hr
        self.ndcg = ndcg 
        self.coverage = coverage


class Tester:

    def __init__(self, algorithm, testloader: torch.utils.data.DataLoader,
      topk: int=10, device: str=None):
        self.algorithm = algorithm
        self.testloader = testloader
        self.topk = topk
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

    @torch.no_grad()
    def __call__(self) -> TestResult:
        hr, ndcg = [], []

        total_recommended_items = set()
        total_items = set()

        for rows in self.testloader:
            total_items.update(rows[:, 1].tolist())
            if self.device:
                rows = rows.to(self.device)
            interactions = rows[:,:2].long()
            real_item = interactions[0][1]
            predictions = self.algorithm(interactions)
            _, indices = torch.topk(predictions, self.topk)
            recommended_items = interactions[indices][:, 1]
            total_recommended_items.update(recommended_items.tolist())
            hr.append(self.__get_hit_ratio(recommended_items, real_item))
            ndcg.append(self.__get_ndcg(recommended_items, real_item))

        cov = len(total_recommended_items) / len(total_items)
        return TestResult(self.topk, mean(hr), mean(ndcg), cov)


class RunTracker:

    HPARAM_PREFIX = "hparam/"

    def __init__(self, hparams: HyperParameters, tb: SummaryWriter=None, embedding_epoch_num=4):
        self.__hparams = hparams
        self.__tb = tb
        self.__metrics = {}
        self.__embedding_md = None
        self.__embedding_md_header = None
        self.__embedding_epoch_num = embedding_epoch_num
        if self.__tb:
            self.__tb.add_text("hparams", str(self.__hparams))

    def setup_embedding(self, idrange: np.ndarray):
        if not self.__tb:
            return
        self.__embedding_md = []
        self.__embedding_md_header = ['label', 'color']
        for i in range(idrange[1]):
            if i < idrange[0]:
                label = f"u{i}"
                color = np.array((1, 0, 0))
            else:
                label = f"i{i}"
                color = np.array((0, 0, 1))
            self.__embedding_md.append((label, color))

    def track_model_epoch(self, epoch: int, model: torch.nn.Module, loss: float, lr: float):
        self.__metrics["hparam/loss"] = loss
        if self.__tb is None:
            return
        self.__tb.add_scalar('train/loss', loss, epoch)
        self.__tb.add_scalar('train/lr', lr, epoch)

        if hasattr(model, 'get_embedding_weight'):
            weight = model.get_embedding_weight()
            if epoch % self.__embedding_epoch_num == 0:
                self.__tb.add_embedding(weight, global_step=epoch,
                    metadata=self.__embedding_md,
                    metadata_header=self.__embedding_md_header)
            self.__tb.add_histogram("embedding", weight, global_step=epoch)
        self.__tb.flush()

    def track_test_result(self, epoch: int, result: TestResult):
        self.__metrics[f"{self.HPARAM_PREFIX}HR"] = result.hr 
        self.__metrics[f"{self.HPARAM_PREFIX}NDCG"] = result.ndcg 
        self.__metrics[f"{self.HPARAM_PREFIX}COV"] = result.coverage
        if self.__tb is None:
            return
        self.__tb.add_scalar(f'eval/HR@{result.topk}', result.hr, epoch)
        self.__tb.add_scalar(f'eval/NDCG@{result.topk}', result.ndcg, epoch)
        self.__tb.add_scalar(f'eval/COV@{result.topk}', result.coverage, epoch)
        self.__tb.flush()

    def track_end(self, run_name=None):
        if self.__tb is None:
            return
        hparams = {}
        for k, v in self.__hparams.data.items():
            hparams[f"{self.HPARAM_PREFIX}{k}"] = v
        self.__tb.add_hparams(hparams, self.__metrics, run_name=run_name)


def create_tensorboard_writer(tb_dir: str, tb_tag: str=None) -> SummaryWriter:
    if not tb_dir:
        return
    if tb_tag:
        tb_dir = os.path.join(tb_dir, tb_tag)
    return SummaryWriter(log_dir=tb_dir)