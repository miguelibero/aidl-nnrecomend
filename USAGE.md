`nnrecommend`
====

## Installation

To install run the following from the root project directory (ideally activate a virtualenv first).

```bash
# replace cu111 with the specific cuda version your machine supports
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install -e ./
```

Once installed make sure you have the python environment scripts directory in your path.

### Datasets

* [`movielens`](https://www.kaggle.com/prajitdatta/movielens-100k-dataset/) movielens dataset from kaggle
* [`podcasts`](https://www.kaggle.com/thoughtvector/podcastreviews) itunes podcaseds dataset from kaggle
* [`spotify`](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge) spotify skip prediction challenge dataset (this is preprocessed data from the dataset)
* [`spotify-mini`](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge) spotify skip prediction challenge mini dataset (can load directly the downloaded files)

### Models

* `fm-linear` factorization machine with linear embedding
* `fm-gcn` factorization machine with graph embedding
* `fm-gcn-att` factorization machine with graph embedding with attention

### Hyper Parameters

* `max_interactions` how many interactions to load from the dataset (`-1` for all)
* `negatives_train` how many negative samples to add to the train dataset (`-1` for all)
* `negatives_test` how many negative samples to add to the test dataset (`-1` for all)
* `batch_size` batch size of the training data loader
* `epochs` amount of epochs to run
* `embed_dim` dimension of the hidden state of the embedding
* `embed_dropout` dropout value for the embedding
* `learning_rate` learning rate of the optimizer
* `lr_scheduler_factor` lr factor for the plateau lr scheduler (`1` by default no scheduler)
* `lr_scheduler_patience` amount of fixed epochs for the plateau lr scheduler
* `lr_scheduler_threshold` threshold for the plateau lr scheduler
* `graph_attention_heads` amount of heads in the GCN with attention model
* `embed_dropout` dropout factor for the embedding
* `pairwise_loss` if we should create the training set with pairs of positive-negative interactions (default `True`)
* `train_loader_workers` amount of workers for the train loader
* `test_loader_workers` amount of workers for the test loader
* `interaction_context` context rows to add, separated by comma (default `all` adds any context)
* `recommend` enable recommend mode

Supported context values are `previous` & `skip`, and they depend on each dataset.

## Command Line Interface

To see the different available actions and parameters run

```bash
nnrecommend --help
```

### Hyperparameters

Passing hyperparameters can be done using `--hparam name:value`, you can add the argument multiple times to set multiple hyper parameters, or `--hparams-path hparams.json` to load the parameters from a json dictionary.

The format of the `hparams.json` file can be a simple dictionary:

```json
{
    "embed_dim": 32,
    "batch_size": 1024
}
```

or a dictionary with trials if you want to run multiple trainings one after the other

```json
{
    "common": {
        "embed_dim": 32
    },
    "trials": [
        {
            "batch_size": 1024
        },
        {
            "batch_size": 512
        }
    ]
}
```

### Training

This command allows you to train a model.

```bash
nnrecommend train --dataset movielens-lab data/ml-dataset-splitted
nnrecommend train --dataset movielens-100k data/ml-100k/
nnrecommend train --dataset podcasts data/database.sqlite
nnrecommend train --dataset spotify data/spotify.csv
```

To select the model:

```bash
nnrecommend train --dataset spotify data/spotify.csv --model fm-gcn
```

To create a tensorboard directory:

```bash
nnrecommend train --dataset spotify data/spotify.csv --tensorboard tbdir
```

Then you can run the tensorboard server on that directory

```bash
tensorboard --logdir tbdir
```


### Fitting

This command allows to fit an algorith with a dataset and get test values.

```bash
nnrecommend fit --dataset spotify data/spotify.csv --algoritm knn --algorithm baseline
```

This command also supports the tensorboard parameter and will create horizontal lines with the test valies for every algorithm.

### Tuning

This command runs hyperparameter tuning with a given dataset and model.
We use [`ray.tune`](https://docs.ray.io/en/master/tune/index.html) for this task.

```bash
nnrecommend tune --dataset spotify data/spotify_2. --model fm-linear --config tune_config.json
```

The command accepts the tune config in a json file with a dictionary with the keys being the hyperparameter names and the values being the `ray.tune` methods that describe the possible values. 
Check the [tune documentation](https://docs.ray.io/en/master/tune/api_docs/search_space.html#tune-sample-docs) for all the possible values.

```json
{
    "learning_rate": ["qloguniform", 1e-4, 1e-1, 5e-4],
    "embed_dropout": ["choice", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
    "batch_size": ["lograndint", 128, 2048],
    "graph_attention_heads": ["randint", 1, 12]
}
```

When running you can see the progress by starting a tensorboard server on the `~/ray_results` folder.

```bash
tensorboard --logdir ~/ray_results
```

### Explore Dataset

This command shows some graphs about the dataset.
It shows for every user, item or context pari:
* histogram of the counts
* spy graph of the adjacency matrix

```
nnrecommend explore-dataset data/ml-100k --type movielens
```

### Recommend

If you train with the `recommend` hyperparameter enabled, the dataset will be modified so that the model trains to recommend items to new users by:
* removing the user column
* creating the previous item context
* switching the items and previous item context columns

If you store the trained module by passing the `--output` parameter, you can use the `recommend` subcommand to get recommendations for new items.

```bash
nnrecommend --hparam recommend:1 --hparam interaction_context: train data/movielens-100k --dataset movielens --output movielens_recommend.pth
nnrecommend recommend movielens_recommend.pth --item "star wars"
```