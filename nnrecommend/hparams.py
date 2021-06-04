
from typing import Dict
import json


class HyperParameters:

    @classmethod
    def fromcli(cls, hparams):
        data = {}
        for hparam in hparams:
            k, v = hparam.split(":", 2)
            data[k] = v
        return cls(data)

    @classmethod
    def fromfile(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))

    DEFAULT_VALUES = {
        "max_interactions": -1,
        "negatives_train": 4,
        "negatives_test": 99,
        "batch_size": 256,
        "epochs": 20,
        "embed_dim": 64,
        "learning_rate": 0.001,
        "scheduler_step_size": 1,
        "scheduler_gamma": 0.99,
        "graph_attention_heads": 4,
        "graph_attention_dropout": 0.5,
    }

    def __init__(self, data: Dict):
        for k, v in self.DEFAULT_VALUES.items():
            if k not in data:
                data[k] = v
            else:
                data[k] = type(v)(data[k])
        self.data = data

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
    def scheduler_step_size(self):
        return self.__get("scheduler_step_size")

    @property
    def scheduler_gamma(self):
        return self.__get("scheduler_gamma")

    @property
    def graph_attention_heads(self):
        return self.__get("graph_attention_heads")

    @property
    def graph_attention_dropout(self):
        return self.__get("graph_attention_dropout")