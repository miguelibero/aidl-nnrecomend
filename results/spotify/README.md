# Model Evaluation using the Spotify Dataset
We compared different models using the [Spotify Skip Predition Challenge](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge-old) as Dataset. We used skip values as context data. If one user skip some song it has a bad ratio and if the song is not skipped it has a better ratio. The tensorboard logdir can be downloaded [here].(https://github.com/miguelibero/aidl-nnrecomend/blob/main/results/movielens/tensorboard.zip)
Some conclusions (Same as movilens):
-   adding the previous item as a context improves the metrics substantially
-   using pairwise loss improves coverage a bit, but it has a lot of memory consumption so we decided not to use pairwise
-   gcn is not much better than linear

### Evaluation
| type | model | context | loss | hit ratio | ndcg | coverage |
| --- | -- | --- | --- | --- | --- | --- |
| nnrecommend | fm-linear |  | ratting | 0.1053 | 0.0902 | 0.1792 | 
| nnrecommend | fm-gcn |  | ratting | 0.1012 | 0.0781 | 0.1952
| nnrecommend | fm-linear | skip | ratting | 0.1071 | 0.0557 | 0.2093
| nnrecommend | fm-gcn | skip | ratting | 0.1495 | 0.0775 | 0.2289
| nnrecommend | fm-linear | previous | ratting | 0.1039 | 0.0540 | 0.2075
| nnrecommend | fm-gcn | previous | ratting | 0.1463 | 0.0736 | 0.2021
| nnrecommend | fm-linear | skip, previous | ratting | 0.1060 | 0.0552 | 0.2105
| nnrecommend | fm-gcn | skip, previous| ratting | 0.1495 | 0.0766 | 0.2241
| nnrecommend | knn |  |  | 0.0891 | 0.0476 | 0.1795

### Hyperparameters

| name | value |
| --- | --- |
| negatives_train | 4 |
| negatives_test | -1 |
| batch_size | 1024 |
| epochs | 40 |
| embed_dim | 64 |
| learning_rate | 0.001 |
| dropout | 0.5 |
| pairwise_loss | 0 |