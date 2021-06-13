
from typing import Any, Dict
import json
from ray import tune


class HyperParameters:

    @classmethod
    def load(cls, hparams: None, path=None):
        data = {}
        try:
            with open(path) as fh:
                data.update(json.load(fh))
        except:
            pass
        if isinstance(hparams, str):
            hparams = hparams.split(";")
        if isinstance(hparams, (tuple, list)):
            for hparam in hparams:
                if isinstance(hparam, str):
                    k, v = hparam.strip().split(":", 2)
                    data[k] = v
        if isinstance(hparams, dict):
            data.update(hparams)
        return cls(data)

    DEFAULT_VALUES = {
        "max_interactions": -1,
        "negatives_train": 4,
        "negatives_test": 99,
        "batch_size": 256,
        "epochs": 20,
        "embed_dim": 64,
        "learning_rate": 0.001,
        "lr_scheduler_patience": 1,
        "lr_scheduler_factor": 0.8,
        "lr_scheduler_threshold": 1e-4,
        "graph_attention_heads": 8,
        "embed_dropout": 0.6,
        "interaction_context": -1
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

    @property
    def max_interactions(self):
        """maximum amount of interactions (dataset will be reduced to this size if bigger)"""
        return self.__get("max_interactions")

    @property
    def negatives_train(self):
        """amount of negative samples to generate for the trainset"""
        return self.__get("negatives_train")

    @property
    def negatives_test(self):
        """amount of negative samples to generate for the testset"""
        return self.__get("negatives_test")

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

    def should_have_interaction_context(self, v: int):
        """
        check if the dataset should add an interaction context
        """
        c = self.__get("interaction_context")
        return c < 0 or c > v 


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