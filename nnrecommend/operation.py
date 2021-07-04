import statistics
import math
import torch
import numpy as np
import os
import tracemalloc
from fuzzywuzzy import process
from logging import Logger
from typing import Any, Callable, Container, Dict
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from nnrecommend.hparams import HyperParameters
from nnrecommend.logging import get_logger
from nnrecommend.dataset import BaseDatasetSource, InteractionDataset, InteractionPairDataset, GroupingDataset, vstack_collate_fn


def human_readable_size(size, decimal_places=2):
    for unit in ['b', 'Kb', 'Mb', 'Gb', 'Tb', 'Pb']:
        if size < 1024.0 or unit == 'Pb':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"


class Setup:

    def __init__(self, src: BaseDatasetSource, logger: Logger=None, trace_memory=False, for_recommend=False):
        self.src = src
        self.__logger = logger or get_logger(self)
        self.__trace_memory = trace_memory
        self.__for_recommend = for_recommend

    def __call__(self, hparams: HyperParameters) -> np.ndarray:

        self.__logger.info(f"using hparams {hparams}")
        self.__logger.info("loading dataset...")

        if self.__trace_memory:
            tracemalloc.start()

        self.src.load(hparams)

        if self.__for_recommend:
            self.__logger.info("preparing for recommend...")
            self.src.trainset.prepare_for_recommend()
            self.src.testset.prepare_for_recommend()

        trainlen, testlen = self.__log_dataset()
        
        idrange = self.src.trainset.idrange
        self.__log_idrange(idrange)
        self.__log_matrix(self.src.matrix)

        trainf = 1.0
        testf = 1.0

        if hparams.negatives_train > 0:
            self.__logger.info("adding trainset negative sampling...")
            matrix = self.src.matrix
            traingroups = self.src.trainset.add_negative_sampling(hparams.negatives_train, matrix)
            trainf = len(self.src.trainset) / trainlen
            if hparams.pairwise_loss:
                self.__logger.info("generating trainset pairs...")
                self.src.trainset = self.__apply_pairs(self.src.trainset, traingroups)

        if hparams.negatives_test:
            self.__logger.info("adding testset negative sampling...")
            testgroups = self.src.testset.add_negative_sampling(hparams.negatives_test, matrix, unique=True)
            testf = len(self.src.testset) / testlen
            self.src.testset = self.__apply_grouping(self.src.testset, testgroups)

        if trainf > 1 or testf > 1:
            self.__logger.info(f"dataset size changed by a factor of {trainf:.2f} train and {testf:.2f} test")

        if self.__trace_memory:
            mem = tracemalloc.get_traced_memory()
            curr_mem, peak_mem = human_readable_size(mem[0]), human_readable_size(mem[1])
            self.__logger.info(f"taking up memory: current={curr_mem} peak={peak_mem}")
            tracemalloc.stop()

        return idrange

    def __apply_grouping(self, dataset: Dataset, groups: np.ndarray):
        return GroupingDataset(dataset, groups)

    def __apply_pairs(self, dataset: Dataset, groups: np.ndarray):
        return InteractionPairDataset(dataset, groups)

    def __log_matrix(self, matrix):
        nnz = matrix.getnnz()
        tot = np.prod(matrix.shape)
        self.__logger.info(f"adjacency matrix {matrix.shape} non-zeros {nnz} ({100*nnz/tot:.4f}%)")

    def __log_dataset(self):
        trainlen = len(self.src.trainset)
        testlen = len(self.src.testset)
        self.__logger.info(f"loaded {trainlen} train and {testlen} test interactions")
        return trainlen, testlen

    def __log_idrange(self, idrange):
        assert len(idrange) >= 2
        lens = []
        lastlen = 0
        for v in idrange:
            lens.append(v - lastlen)
            lastlen = v
        if len(lens) == 2:
            self.__logger.info(f"loaded {lens[0]} users and {lens[1]} items")
        else:
            clens = "/".join([str(v) for v in lens[2:]])
            self.__logger.info(f"loaded {lens[0]} users, {lens[1]} items and {clens} contexts")

    def create_testloader(self, hparams: HyperParameters):
        dataset = self.src.testset
        return DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=vstack_collate_fn, num_workers=hparams.test_loader_workers)

    def create_trainloader(self, hparams: HyperParameters):
        dataset = self.src.trainset
        return DataLoader(dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.train_loader_workers)


class Trainer:

    def __init__(self, model: torch.nn.Module, trainloader: DataLoader,
            optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: str=None):
        self.model = model
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def __forward(self, rows):
        if self.device:
            rows = rows.to(self.device)
        interactions = rows[:,:-1].long()
        targets = rows[:,-1].float()
        predictions = self.model(interactions)
        return predictions, targets

    def __call__(self) -> float:
        """
        run a training epoch
        :return: the mean loss
        """
        total_loss = []

        for batch in self.trainloader:
            self.optimizer.zero_grad()
            
            if isinstance(batch, Tensor):
                predictions, targets = self.__forward(batch)
                loss = self.criterion(predictions, targets)
            elif isinstance(batch, (list, tuple)):
                predpos, _ = self.__forward(batch[0])
                predneg, _ = self.__forward(batch[1])
                loss = self.criterion(predpos, predneg)
            else:
                raise ValueError("batch")

            loss.backward()
            self.optimizer.step()
            total_loss.append(loss.item())

        return statistics.mean(total_loss)


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

    def __str__(self):
        return f"hr={self.hr:.4f} ndcg={self.ndcg:.4f} cov={self.coverage:.4f}"


class Tester:

    def __init__(self, model: Callable, testloader: torch.utils.data.DataLoader,
      topk: int=10, device: str=None):
        self.model = model
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

        for batch in self.testloader:
            if batch is None or batch.shape[0] == 0:
                continue
            total_items.update(batch[:, 1].tolist())
            if self.device:
                batch = batch.to(self.device)
            interactions = batch[:,:-1].long()
            real_item = interactions[0][1]
            predictions = self.model(interactions)
            _, indices = torch.topk(predictions, self.topk)
            recommended_items = interactions[indices][:, 1]
            total_recommended_items.update(recommended_items.tolist())
            hr.append(self.__get_hit_ratio(recommended_items, real_item))
            ndcg.append(self.__get_ndcg(recommended_items, real_item))

        cov = len(total_recommended_items) / len(total_items)
        return TestResult(self.topk, statistics.mean(hr), statistics.mean(ndcg), cov)


class RunTracker:

    HPARAM_PREFIX = "hparam/"
    COLORS = (
        (1, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
    )

    def __init__(self, hparams: HyperParameters, tb: SummaryWriter=None, embedding_epoch_num=False):
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
        last = 0
        for i, current in enumerate(idrange):
            for j in range(last, current):
                label = f"{i}_{j}"
                color = self.COLORS[i % len(self.COLORS)]
                self.__embedding_md.append((label, color))
            last = current

    def track_model_epoch(self, epoch: int, model: torch.nn.Module, loss: float, lr: float):
        self.__metrics["hparam/loss"] = loss
        if self.__tb is None:
            return
        self.__tb.add_scalar('train/loss', loss, epoch)
        self.__tb.add_scalar('train/lr', lr, epoch)

        if hasattr(model, 'get_embedding_weight'):
            weight = model.get_embedding_weight()
            if self.__embedding_epoch_num > 0 and epoch % self.__embedding_epoch_num == 0:
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
        # disabled since it does not look to good in tensorboard
        # self.__tb.add_hparams(hparams, self.__metrics, run_name=run_name)


class FinderResult:
    def __init__(self, id: int, field: str, value: str, score: int):
        self.id = id
        self.field = field
        self.value = value
        self.score = score

    def __str__(self):
        return f"FinderResult(id={self.id}, field={self.field}, value={self.value}, score={self.score})"


class Finder:
    """
    given an info dictionary finds the element that matches a string best
    info dictionary should be in the form of:

    {
        id1: {
            "field_name": "field value",
            "other_field_name": "other field value",
        },
        id2: {
            "field_name": "field value",
            "other_field_name": "other field value",
        },
        ...
    }
    """

    def __init__(self, info: Dict[int, Dict[str, Any]], fields: Container[str]=None):
        self.__fields = {}
        assert isinstance(info, dict)
        for id, elm in info.items():
            assert isinstance(elm, dict)
            for k, v in elm.items():
                if fields is not None and k not in fields:
                    continue
                if k not in self.__fields:
                    f = {}
                    self.__fields[k] = f
                else:
                    f = self.__fields[k]
                f[id] = v

    def __call__(self, v: str) -> FinderResult:
        best = None
        for name, f in self.__fields.items():
            r = process.extractOne(v, f)
            if best is None or best[1] < r[1]:
                best = r + (name,)
        return FinderResult(best[2], best[3], best[0], best[1])


def create_tensorboard_writer(tb_dir: str, tb_tag: str=None) -> SummaryWriter:
    if not tb_dir:
        return
    if tb_tag:
        tb_dir = os.path.join(tb_dir, tb_tag)
    return SummaryWriter(log_dir=tb_dir)