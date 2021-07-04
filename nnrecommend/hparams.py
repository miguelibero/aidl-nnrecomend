
from typing import Any, Dict
from ray import tune
import json


class HyperParameters:

    TRIALS_KEY = "trials"
    COMMON_KEY = "common"

    @classmethod
    def load_trials(cls, cmdargs: None, path=None):
        data = {}
        try:
            with open(path) as fh:
                data.update(json.load(fh))
        except:
            pass

        if isinstance(cmdargs, str):
            cmdargs = cmdargs.split(";")

        if cls.TRIALS_KEY not in data:
            if not isinstance(data, (list, tuple)):
                data = (data,)
            data = {cls.TRIALS_KEY: data}
        trials = []
        common = data[cls.COMMON_KEY] if cls.COMMON_KEY in data else {}
        for trial in data[cls.TRIALS_KEY]:
            trial.update(common)
            if isinstance(cmdargs, (tuple, list)):
                for hparam in cmdargs:
                    if isinstance(hparam, str):
                        k, v = hparam.strip().split(":", 2)
                        trial[k] = v
            elif isinstance(cmdargs, dict):
                trial.update(cmdargs)
            trials.append(cls(trial))
        return trials


    DEFAULT_VALUES = {
        "max_interactions": -1,
        "negatives_train": 10,
        "negatives_test": 99,
        "batch_size": 1024,
        "epochs": 40,
        "embed_dim": 64,
        "learning_rate": 0.01,
        "lr_scheduler_patience": 1,
        "lr_scheduler_factor": 0.8,
        "lr_scheduler_threshold": 1e-4,
        "graph_attention_heads": 8,
        "embed_dropout": 0.5,
        "interaction_context": "all",
        "pairwise_loss": True,
        "train_loader_workers": 0,
        "test_loader_workers": 0,
        "previous_items_cols": 1
    }

    def __init__(self, data: Dict = {}):
        for k, v in self.DEFAULT_VALUES.items():
            if k not in data:
                data[k] = v
            elif v is not None:
                data[k] = type(v)(data[k])
        self.data = data

    def copy(self, data: Dict=None):
        cdata = self.data.copy()
        if data:
            cdata.update(data)
        return __class__(cdata)

    def __str__(self):
        data = " ".join([f"{k}={v}" for k, v in self.data.items()])
        return "{" + data + "}"

    def __get(self, key):
        if key in self.data:
            return self.data[key]

    def __set(self, key, val):
        self.data[key] = val

    @property
    def max_interactions(self):
        """maximum amount of interactions (dataset will be reduced to this size if bigger)"""
        return self.__get("max_interactions")

    @property
    def negatives_train(self):
        """amount of negative samples to generate for the trainset"""
        return self.__get("negatives_train")

    @negatives_train.setter
    def negatives_train(self, v):
        return self.__set("negatives_train", v)

    @property
    def negatives_test(self):
        """
        amount of negative samples to generate for the testset
        negative or non means it will add all the possible item values
        """
        return self.__get("negatives_test")

    @negatives_test.setter
    def negatives_test(self, v):
        return self.__set("negatives_test", v)

    @property
    def batch_size(self):
        """batchsize of the trainset dataloader"""
        return self.__get("batch_size")

    @property
    def epochs(self):
        # TODO: check if this should be a hyper parameter or fixed
        """amount of epochs to run the training"""
        return self.__get("epochs")

    @property
    def embed_dim(self):
        """size of the embedding state"""
        return self.__get("embed_dim")

    @property
    def learning_rate(self):
        """learning rate"""
        return self.__get("learning_rate")

    @property
    def lr_scheduler_patience(self):
        """
        patience parameter of torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        return self.__get("lr_scheduler_patience")

    @property
    def lr_scheduler_factor(self):
        """
        factor parameter of torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        return self.__get("lr_scheduler_factor")

    @property
    def lr_scheduler_threshold(self):
        """
        threshold parameter of torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        return self.__get("lr_scheduler_threshold")

    @property
    def embed_dropout(self):
        """
        dropout vale for the embedding module
        """
        return self.__get("embed_dropout")

    @property
    def graph_attention_heads(self):
        """
        amount of heads in the GCN with attention
        """
        return self.__get("graph_attention_heads")

    @property
    def pairwise_loss(self):
        """
        train the model using pairwise loss
        """
        return self.__get("pairwise_loss")

    @pairwise_loss.setter
    def pairwise_loss(self, value: bool):
        self.__set("pairwise_loss", bool(value))

    @property
    def train_loader_workers(self):
        """
        amount of workers to use for the train_loader_workers DataLoader
        """
        return self.__get("train_loader_workers")

    @property
    def test_loader_workers(self):
        """
        amount of workers to use for the test DataLoader
        """
        return self.__get("test_loader_workers")

    def get_tensorboard_tag(self, defval: str, **kwargs) -> str:
        """
        get the tensorboard tag
        """
        tag = self.__get("tensorboard_tag") or defval
        return tag.format(
            tag=defval,
            **kwargs
        )

    @property
    def interaction_context(self):
        """
        list of contexts to put in the dataset
        """
        return self.__get("interaction_context")

    @interaction_context.setter
    def interaction_context(self, val: str):
        return self.__set("interaction_context", str(val))

    @property
    def previous_items_cols(self):
        """
        the amount of previous items columns to add
        """
        if not self.should_have_interaction_context("previous"):
            return 0
        return self.__get("previous_items_cols")

    @previous_items_cols.setter
    def previous_items_cols(self, val: int):
        return self.__set("previous_items_cols", int(val))

    def should_have_interaction_context(self, v: str):
        """
        check if the dataset should add an interaction context
        """
        parm = self.__get("interaction_context")
        if not parm:
            return False
        if parm == "all":
            return True
        v = str(v)
        if not isinstance(parm, (list, tuple)):
            parm = str(parm).split(",")
        return v in parm



class RayTuneConfigFile:

    @classmethod
    def load(cls, path=None):
        with open(path) as fh:
            return cls(json.load(fh))

    def __init__(self, data=Dict[str, Any]):
        self.data = data

    def generate(self, model_type=None):
        config = {}
        for k, v in self.data.items():
            try:
                config[k] = getattr(tune, v[0])(*v[1:])
            except:
                config[k] = v
        return config