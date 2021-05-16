from nnrecommend.dataset import Dataset


def test_dataset():
    data = ((2, 2), (3, 1))
    dataset = Dataset(data)
    assert len(dataset) == 2
    assert (dataset[0] == (2, 2, 1)).all()
    assert (dataset[1] == (3, 1, 1)).all()
    assert dataset.idrange == None
    iddiff = dataset.normalize_ids()
    assert len(dataset) == 2
    assert (dataset[0] == (0, 3, 1)).all()
    assert (dataset[1] == (1, 2, 1)).all()
    assert (dataset.idrange == (2, 4)).all()
    assert (iddiff == (-2, 1)).all()


def test_dataset_iddiff():
    data = ((2, 2), (3, 1))
    dataset = Dataset(data)
    iddiff = dataset.normalize_ids((-3, 5))
    assert (iddiff == (-3, 5)).all()
    assert (dataset[0] == (-1, 7, 1)).all()
    assert (dataset[1] == (0, 6, 1)).all()
    assert (dataset.idrange == (1, 8)).all()


def test_adjacency_matrix():
    data = ((2, 2), (3, 1))
    dataset = Dataset(data)
    matrix = dataset.create_adjacency_matrix()
    assert (0, 3) in matrix
    assert (0, 2) not in matrix
    assert (1, 2) in matrix
    nitem = dataset.get_random_negative_item(0, 3, matrix)
    assert nitem == 2


def test_negative_sampling():
    data = ((2, 2), (3, 1))
    dataset = Dataset(data)
    matrix = dataset.create_adjacency_matrix()
    dataset.add_negative_sampling(matrix, 2)
    assert len(dataset) == 6
    assert (dataset[0] == (0, 3, 1)).all()
    assert (dataset[1] == (0, 2, 0)).all()
    assert (dataset[2] == (0, 2, 0)).all()
    assert (dataset[3] == (1, 2, 1)).all()
    assert (dataset[4] == (1, 3, 0)).all()
    assert (dataset[5] == (1, 3, 0)).all()


def test_extract_test_dataset():
    data = ((2, 2), (2, 3), (3, 1), (3, 4))
    dataset = Dataset(data)
    matrix = dataset.create_adjacency_matrix()
    dataset.add_negative_sampling(matrix, 2)
    assert len(dataset) == 12
    testset = dataset.extract_test_dataset()
    assert type(testset) == Dataset
    assert len(dataset) == 10
    assert len(testset) == 2