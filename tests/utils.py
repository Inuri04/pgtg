import numpy as np


def assert_observations_equal(observation_1: dict, observation_2: dict) -> None:
    assert np.array_equal(observation_1["position"], observation_2["position"])
    assert np.array_equal(observation_1["velocity"], observation_2["velocity"])
    assert observation_1["map"].keys() == observation_2["map"].keys()
    for key in observation_1["map"].keys():
        assert np.array_equal(observation_1["map"][key], observation_2["map"][key])
