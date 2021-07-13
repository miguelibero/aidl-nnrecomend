import statistics
import math
from pandas.core.frame import DataFrame
import torch
import numpy as np
import os
import tracemalloc
from fuzzywuzzy import process
from logging import Logger
from typing import Any, Callable, Container, Dict
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nnrecommend.hparams import HyperParameters
from nnrecommend.logging import get_logger
from nnrecommend.dataset import BaseDatasetSource, InteractionPairDataset, GroupingDataset, vstack_collate_fn


def human_readable_size(size, decimal_places=2):
    for unit in ['b', 'Kb', 'Mb', 'Gb', 'Tb', 'Pb']:
        if size < 1024.0 or unit == 'Pb':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"


def freemem():
    # tested using this to reduce CUDA out of memory errors, rn it's disabled
    # if torch.cuda.is_available():
    #    torch.cuda.empty_cache()
    pass


class Setup:
    def __init__(self, src: BaseDatasetSource, logger: Logger=None, trace_memory=False):
        self.src = src
        self._logger = logger or get_logger(self)
        self.__trace_memory = trace_memory
        self.__pairstrainset = None
        self.__groupstestset = None

    def __log_dataset(self):
        trainlen = len(self.src.trainset)
        testlen = len(self.src.testset)
        self._logger.info(f"loaded {trainlen} train and {testlen} test interactions")
        return trainlen, testlen

    def __log_idrange(self, idrange, hparams: HyperParameters):
        assert len(idrange) >= 2
        lens = []
        lastlen = 0
        if hparams.recommend:
            colnames = ('previous items', 'items')
        else:
            colnames = ('users', 'items')

        for v in idrange:
            lens.append(v - lastlen)
            lastlen = v
        if len(lens) == 2:
            self._logger.info(f"loaded {lens[0]} {colnames[0]} and {lens[1]} {colnames[1]}")
        else:
            clens = "/".join([str(v) for v in lens[2:]])
            self._logger.info(f"loaded {lens[0]} {colnames[0]}, {lens[1]} {colnames[1]} and {clens} contexts")

    def __call__(self, hparams: HyperParameters) -> np.ndarray:

        self._logger.info(f"using hparams {hparams}")
        self._logger.info("loading dataset...")

        if self.__trace_memory:
            tracemalloc.start()

        idrange = self._load(hparams)

        if self.__trace_memory:
            mem = tracemalloc.get_traced_memory()
            curr_mem, peak_mem = human_readable_size(mem[0]), human_readable_size(mem[1])
            self._logger.info(f"taking up memory: current={curr_mem} peak={peak_mem}")
            tracemalloc.stop()

        return idrange

    def get_items(self):
        return self.src.items

    def _load(self, hparams: HyperParameters) -> np.ndarray:
        self.src.load(hparams)
        self.__log_dataset()
        idrange = self.src.trainset.idrange
        self.__log_idrange(idrange, hparams)

        trainf, testf = 1.0, 1.0

        self._logger.info("adding trainset negative sampling...")
        trainlen = len(self.src.trainset)
        traingroups = self.src.trainset.add_negative_sampling(hparams.negatives_train, self.src.useritems)
        trainf = len(self.src.trainset) / trainlen
        if hparams.pairwise_loss:
            self._logger.info("generating trainset pairs...")
            self.__pairstrainset = InteractionPairDataset(self.src.trainset, traingroups)

        self._logger.info("adding testset negative sampling...")
        testlen = len(self.src.testset)
        testgroups = self.src.testset.add_negative_sampling(hparams.negatives_test, self.src.useritems, unique=True)
        testf = len(self.src.testset) / testlen
        self.__groupstestset = GroupingDataset(self.src.testset, testgroups)

        if trainf > 1 or testf > 1:
            self._logger.info(f"dataset size changed by a factor of {trainf:.2f} train and {testf:.2f} test")

        return idrange

    def create_testloader(self, hparams: HyperParameters):
        if self.__groupstestset:
            dataset = self.__groupstestset
            batch_size = 1
            collate = vstack_collate_fn
        else:
            dataset = self.src.testset
            batch_size = hparams.batch_size
            collate = None
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=hparams.test_loader_workers)

    def create_trainloader(self, hparams: HyperParameters):
        dataset = self.__pairstrainset if self.__pairstrainset else self.src.trainset
        return DataLoader(dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.train_loader_workers)

    def create_adjacency_matrix(self, hparams: HyperParameters):
        self._logger.info("creating adjacency matrix...")
        matrix = self.src.trainset.create_adjacency_matrix()
        self.__log_matrix(matrix)
        return matrix

    def __log_matrix(self, matrix):
        nnz = matrix.getnnz()
        tot = np.prod(matrix.shape)
        self._logger.info(f"adjacency matrix {matrix.shape} non-zeros {nnz} ({100*nnz/tot:.4f}%)")


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
        freemem()

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

            del batch
            freemem()
            loss.backward()
            self.optimizer.step()
            if not torch.isnan(loss):
                total_loss.append(loss.item())

        if len(total_loss) == 0:
            return 0.0
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

    def to_dict(self) -> Dict[str, str]:
        return {
            "HR": self.hr,
            "NDCG": self.ndcg,
            "COV": self.coverage
        }

    def __gt__(self, result: 'TestResult') -> bool:
        return self.ndcg > result.ndcg

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
        freemem()

        for batch in self.testloader:
            if batch is None or batch.shape[0] == 0:
                continue
            total_items.update(batch[:, 1].tolist())
            if self.device:
                batch = batch.to(self.device)
            interactions = batch[:, :-1].long()
            real_item = interactions[0][1]
            predictions = self.model(interactions)
            _, indices = torch.topk(predictions, self.topk)
            recommended_items = interactions[indices][:, 1]
            total_recommended_items.update(recommended_items.tolist())
            hr.append(self.__get_hit_ratio(recommended_items, real_item))
            ndcg.append(self.__get_ndcg(recommended_items, real_item))
            del batch
            freemem()

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
            if weight is not None and len(weight) > 0:
                if self.__embedding_epoch_num > 0 and epoch % self.__embedding_epoch_num == 0:
                    self.__tb.add_embedding(weight, global_step=epoch,
                        metadata=self.__embedding_md,
                        metadata_header=self.__embedding_md_header)
                self.__tb.add_histogram("embedding", weight, global_step=epoch)
        self.__tb.flush()

    def track_test_result(self, epoch: int, result: TestResult):
        rdict = result.to_dict()
        for k, v in rdict.items():
            self.__metrics[f"{self.HPARAM_PREFIX}{k}"] = v 
        if self.__tb is None:
            return
        for k, v in rdict.items():
            self.__tb.add_scalar(f'eval/{k}@{result.topk}', v, epoch)
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
        self.id = int(id)
        self.field = str(field)
        self.value = str(value)
        self.score = int(score)

    def __str__(self):
        return f"FinderResult(id={self.id}, field={self.field}, value={self.value}, score={self.score})"


class Finder:
    def __init__(self, items: DataFrame, fields: Container[str]=None):
        assert isinstance(items, DataFrame)
        self.items = items
        self.fields = fields

    def __call__(self, v: str) -> FinderResult:
        try:
            v = int(v)
            item = self.items.loc[v]
            return FinderResult(v, "id", v, -1)
        except ValueError:
            pass
        best = None
        columns = self.fields or self.items.select_dtypes(include=object)
        for colname in columns:
            f = self.items[colname]
            r = process.extractOne(v, f)
            if best is None or best[1] < r[1]:
                best = r + (colname,)
        # id, field, value, score
        if best:
            return FinderResult(best[2], best[3], best[0], best[1])


class Recommender:

    def __init__(self, idrange: np.ndarray, items: DataFrame, model: Callable, device: str=None):
        self.items = items
        self.model = model
        self.device = device
        self.idrange = idrange

    def __create_input(self, ids: Container[int], remove_itemids: Container[int]=None) -> np.ndarray:
        """
        create an array with the following columns
        * the last id in the container
        * all possible items (item column that will be evaluated)
        * the rest of ids in reverse order
        """
        if len(ids) + 1 != len(self.idrange):
            raise ValueError("invalid amount of ids")

        itemids = np.array(self.items.index)
        if remove_itemids is not None:
            itemids = np.delete(itemids, remove_itemids)
        data = np.zeros((len(itemids), len(self.idrange)), dtype=np.int64)
        data[:, 0] = ids[-1]
        data[:, 1] = itemids + self.idrange[0]
        for i in range(2, data.shape[1]):
            data[:, i] = ids[-1*i] + self.idrange[i-1]

        # check that the ranges are ok
        minv = 0
        for i, maxv in enumerate(self.idrange):
            col = data[:, i]
            if not np.logical_and(col >= minv, col <= maxv).all():
                raise ValueError(f"invalid value in column {i}")
            minv = maxv

        data = torch.from_numpy(data)
        if self.device:
            data = data.to(self.device)
        return data

    def __call__(self, ids: Container[int], topk: int=3, remove_ids: Container[int]=None):
        input = self.__create_input(ids, remove_ids)
        predictions = self.model(input)
        ratings, indices = torch.topk(predictions, topk)
        itemids = input[indices][:, 1] - self.idrange[0]
        del input
        ratings = ratings.cpu().tolist()
        itemids = itemids.cpu().tolist()
        for rid, rating in zip(itemids, ratings):
            row = self.items.loc[rid]
            yield row, rating


def create_tensorboard_writer(tb_dir: str, tb_tag: str=None) -> SummaryWriter:
    if not tb_dir:
        return
    if tb_tag:
        tb_dir = os.path.join(tb_dir, tb_tag)
    return SummaryWriter(log_dir=tb_dir)
