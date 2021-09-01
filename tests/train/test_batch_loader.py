import pytest

from numpy import array

from deep_attribution.train.batch_loader import BatchLoader
from deep_attribution.model.journey_based_deepnn import JourneyBasedDeepNN

def MOCK_LOAD_BATCH(self, train_file_nm):
            
    return [
        array([
            [0, 0, 1, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0]
        ]),
        array([True, False])
    ]

def MOCK_GET_BATCH_FILE_NMS(self):

    return [1, 2, 3] # BatchLoader.__load_batch() is mocked here, batch 
                               # file nms only set len

def test_next(mocker):

    SET_NM = "train"
   
    NB_CAMPAIGNS = 3
    JOURNEY_MAX_LEN = 3

    EXPECTED = [
        array([
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[1, 0, 0], [0, 0, 1], [0, 0, 0]]
        ]),
        array([True, False])
    ]
    NB_BATCHS = 3

    mocker.patch(
        "deep_attribution.train.batch_loader.BatchLoader._BatchLoader__get_batch_file_nms",
        MOCK_GET_BATCH_FILE_NMS
        )
    mocker.patch(
        "deep_attribution.train.batch_loader.BatchLoader._BatchLoader__load_batch",
        MOCK_LOAD_BATCH
        )

    batch_loader = BatchLoader(
        "conversion_status",
        SET_NM,
        NB_CAMPAIGNS,
        JOURNEY_MAX_LEN,
        set_parent_dir_path = "" #not used because of mock
    )

    assert NB_BATCHS == len(batch_loader)
    
    batch_iterator = iter(batch_loader)

    for X, y in batch_iterator:
        assert (EXPECTED[0] == X).all() and (EXPECTED[1] == y).all()


def test_model_integration(mocker):

    SET_NM = "test" #skip oversample
    LOAD_BATCH_OUTPUT = [
        array([
            [0, 0, 1, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0]
        ]),
        array([True, False])
    ]
    BATCH_FILE_NMS = [1, 2, 3] # BatchLoader.__load_batch() is mocked here, batch 
                               # file nms only set len
    NB_CAMPAIGNS = 3
    JOURNEY_MAX_LEN = 3

    mocker.patch(
        "deep_attribution.train.batch_loader.BatchLoader._BatchLoader__get_batch_file_nms",
        MOCK_GET_BATCH_FILE_NMS
        )

    mocker.patch(
        "deep_attribution.train.batch_loader.BatchLoader._BatchLoader__load_batch",
        MOCK_LOAD_BATCH
        )
        
    batch_loader = BatchLoader(
        "conversion_status",
        SET_NM,
        NB_CAMPAIGNS,
        JOURNEY_MAX_LEN,
        set_parent_dir_path=""
        
    )

    model = JourneyBasedDeepNN(
        JOURNEY_MAX_LEN,
        NB_CAMPAIGNS
    )

    model.batch_fit(batch_loader, batch_loader)

    assert True

