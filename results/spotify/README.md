# Model Evaluation using the Spotify Dataset
We compared different models using the [Spotify Skip Predition Challenge](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge-old) as Dataset. We used skip values as context data. If one user skip some song it has a bad ratio and if the song is not skipped it has a better ratio. The tensorboard logdir can be downloaded [here](https://github.com/miguelibero/aidl-nnrecomend/blob/main/results/spotify/tensorboard.zip).
Some conclusions (Same as movilens):
-   adding the previous item as a context improves the metrics substantially
-   using pairwise loss improves coverage a little bit, but it has a lot of memory consumption so we decided not to use pairwise
-   gcn is not much better than linear

### Evaluation
| type | model | context | loss | hit ratio | ndcg | coverage |
| --- | -- | --- | --- | --- | --- | --- |
| nnrecommend | fm-linear |  | 0.1053 | 0.1644 | 0.0902 | 0.1792 | 
| nnrecommend | fm-gcn |  | 0.1022 | 0.3154 | 0.1952 | 0.2630 |
| nnrecommend | fm-linear | skip | 0.2475 | 0.0515 | 0.0266 | 0.0239 |
| nnrecommend | fm-gcn | skip | 0.3358 | 0.0519 | 0.0281 | 0.0025 |
| nnrecommend | fm-linear | previous | 0.1025 | 0.1777 | 0.1192 | 0.1344 |
| nnrecommend | fm-gcn | previous | 0.1092 | 0.3569 | 0.2226 | 0.2391 |
| nnrecommend | fm-linear | skip, previous | 0.1614 | 0.1050 | 0.0675 | 0.0557 |
| nnrecommend | fm-gcn | skip, previous | 0.2555 | 0.1290 | 0.0712 | 0.0304 |
| nnrecommend | knn |  |  |  |  |  |  



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
