`nnrecommend`
====

The recommender is a valid python package.

## Command Line Interface

To install run the following from the root project directory (ideally activate a virtualenv first).

```bash
# replace cu111 with the specific cuda version your machine supports
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install -e ./
```

To see the different available actions and parameters run

```bash
nnrecommend --help
```

### Movielens

Download from [here](https://drive.google.com/uc?id=1rE20sLow9sT2ULpBOOWqw2SEnpIm16OZ) and uncompress

```bash
nnrecommend train --dataset movielens ./path/to/ml-dataset-splitted
```